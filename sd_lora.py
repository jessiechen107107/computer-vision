"""Train an SDXL UNet LoRA and run the LoRA experiment's inference commands.

Running this file with no arguments trains LoRA as needed, then generates and
scores the full test set. For explicit CLI commands, only ``train`` changes
model weights; ``all`` generates and evaluates an already saved adapter.
Generation, full-caption text encoding, CLIPScore, and FID are implemented in
sd.py so the original SDXL and LoRA groups use the same experimental protocol.

Authoring status: this source was written and reviewed as text only. It has not
been imported, compiled, executed, or tested, as requested by the project owner.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import uuid

from data import (
    EpochSampler,
    FoodDataset,
    collate_training_batch,
    file_sha256,
    json_sha256,
    load_split,
    manifest_sha,
    resolve_path,
    write_json,
)
from sd import (
    ENCODING_VERSION,
    adapter_identity,
    build_pipeline,
    compute_clipscore,
    encode_caption,
    exclusive_file,
    generate_images,
    inference_commands,
    load_config,
    model_lock,
    report_when_ready,
    run_root,
    runtime_device_dtype,
    save_run_metadata,
    set_seed,
)


ADAPTER_NAME = "food500cap"
CHECKPOINT_VERSION = 2


def training_root(config: dict) -> Path:
    """Keep this group's training artifacts separate from baseline outputs."""
    return run_root(config) / "sd_lora"


def _check_training_config(config: dict) -> None:
    """Reject unsupported settings before allocating several large models.

    Per-device batch size is deliberately one. Padding a short caption to match
    a longer caption adds extra CLIP windows and would make its conditioning
    depend on its batch partners. Accumulation and DDP provide larger effective
    batches without changing the shared full-caption encoding convention.
    """
    for section in ("train", "runtime", "text", "generation", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")
    train = config["train"]
    for key in ("resolution", "lora_rank", "batch_size_per_gpu", "gradient_accumulation_steps",
                "max_train_steps", "checkpoint_every_steps", "validate_every_steps", "validation_samples"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    for key in ("num_workers", "warmup_steps"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"train.{key} must be a nonnegative integer")
    for section, key in (("experiment", "train_seed"), ("train", "validation_seed")):
        if isinstance(config[section].get(key), bool) or not isinstance(config[section].get(key), int):
            raise ValueError(f"{section}.{key} must be an integer")
    for key in ("lora_alpha", "learning_rate", "adam_epsilon", "max_grad_norm"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"train.{key} must be finite and positive")
    for key in ("adam_beta1", "adam_beta2"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"train.{key} must be in [0, 1)")
    decay = train.get("adam_weight_decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
        raise ValueError("train.adam_weight_decay must be finite and nonnegative")
    modules = train.get("target_modules")
    if not isinstance(modules, list) or not modules or any(not isinstance(name, str) or not name for name in modules):
        raise ValueError("train.target_modules must be a nonempty list of module names")
    if len(set(modules)) != len(modules):
        raise ValueError("train.target_modules contains duplicates")
    if not isinstance(train.get("gradient_checkpointing"), bool):
        raise ValueError("train.gradient_checkpointing must be true or false")
    if train["batch_size_per_gpu"] != 1:
        raise ValueError("Full-caption conditioning requires train.batch_size_per_gpu=1; use gradient accumulation instead")
    resolution = train["resolution"]
    if resolution < 8 or resolution % 8:
        raise ValueError("train.resolution must be a positive multiple of eight")
    if config["generation"].get("height") != resolution or config["generation"].get("width") != resolution:
        raise ValueError("generation.height and generation.width must both equal train.resolution")
    if config["text"].get("long_caption_mode") != "chunk":
        raise ValueError("Training requires the shared full-caption chunk encoding")
    if not isinstance(config["generation"].get("negative_prompt", ""), str):
        raise ValueError("generation.negative_prompt must be a string")


def _main_process_call(accelerator, function, description: str):
    """Let every rank receive the result or a useful rank-zero failure.

    Simply raising on rank zero while other ranks wait at a barrier can leave
    a DDP job hanging. Small control-plane operations therefore broadcast their
    success or failure before the ranks continue. Tensor communication remains
    Accelerate's responsibility.
    """
    from accelerate.utils import broadcast_object_list

    message = [None]
    if accelerator.is_main_process:
        try:
            message[0] = {"ok": True, "value": function()}
        except Exception as error:
            message[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    broadcast_object_list(message, from_process=0)
    if not message[0]["ok"]:
        raise RuntimeError(f"{description} failed on the main process: {message[0]['error']}")
    return message[0]["value"]


@contextmanager
def _preserve_training_rng():
    """Make validation invisible to subsequent training random-number draws.

    The generator used by shared image generation is already local to each
    sample. Saving every process RNG also covers evaluator/model construction
    and library internals that may draw from a global generator.
    """
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _strict_rng_name(process_index: int) -> str:
    return f"strict_rng_rank_{process_index:05d}.json"


def _save_strict_rng(accelerator, directory: Path) -> None:
    """Save a small, non-pickle RNG sidecar for each training process.

    Accelerate also saves RNGs, but its pinned loader logs and suppresses some
    RNG restoration failures. An optimizer can then resume while random draws
    silently restart from setup seeds. These independently hashed JSON states
    make that failure explicit and provide strict restoration below.
    """
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if accelerator.device.type == "cuda" else []
    write_json(directory / _strict_rng_name(accelerator.process_index), {
        "process_index": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [state.cpu().tolist() for state in cuda_states],
    })


def _restore_strict_rng(accelerator, directory: Path) -> None:
    """Restore all training RNGs explicitly; any error aborts the resume.

    JSON lists are reconstructed as the tuple/uint32/ByteTensor representations
    expected by Python, NumPy, and PyTorch. No executable pickle is needed for
    this additional confirmation. The visible CUDA-device count must match the
    saved process topology so no saved generator can be silently ignored.
    """
    import numpy as np
    import torch

    name = _strict_rng_name(accelerator.process_index)
    completion = json.loads((directory / "checkpoint_complete.json").read_text(encoding="utf-8"))
    expected_hash = completion.get("strict_rng_sha256", {}).get(name)
    if expected_hash is None or file_sha256(directory / name) != expected_hash:
        raise ValueError(f"Missing or corrupt strict RNG state: {directory / name}")
    saved = json.loads((directory / name).read_text(encoding="utf-8"))
    if saved["process_index"] != accelerator.process_index or saved["world_size"] != accelerator.num_processes:
        raise ValueError("Strict RNG state belongs to a different process topology")
    expected_cuda = torch.cuda.device_count() if accelerator.device.type == "cuda" else 0
    if len(saved["torch_cuda"]) != expected_cuda:
        raise ValueError("Resume requires the same visible CUDA-device count as the saved RNG state")
    python_state = saved["python"]
    random.setstate((int(python_state[0]), tuple(python_state[1]), python_state[2]))
    numpy_state = saved["numpy"]
    np.random.set_state((
        numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]), int(numpy_state["has_gauss"]), float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(torch.tensor(saved["torch_cpu"], dtype=torch.uint8, device="cpu"))
    if expected_cuda:
        torch.cuda.set_rng_state_all([
            torch.tensor(values, dtype=torch.uint8, device="cpu") for values in saved["torch_cuda"]
        ])


def _training_signature(config: dict, base_lock: dict, clip_lock: dict,
                        accelerator, precision: str, validation_records: list[dict]) -> dict:
    """Bind resume to the exact training problem and distributed layout.

    An optimizer state is meaningful only for the same adapters, data order,
    accumulation policy, schedule, and base weights. The complete public config
    is retained rather than maintaining a fragile list of selected fields.
    Absolute config-location helper keys are excluded so a job may be resumed
    from a relocated but otherwise identical experiment folder.
    """
    source_root = Path(__file__).resolve().parent
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "diffusers", "transformers", "peft", "accelerate", "safetensors")}
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "base_model": base_lock,
        "clip_model": clip_lock,
        "train_manifest_sha256": manifest_sha(config, "train"),
        "val_manifest_sha256": manifest_sha(config, "val"),
        "validation_sample_ids": [record["sample_id"] for record in validation_records],
        "encoding_version": ENCODING_VERSION,
        "world_size": accelerator.num_processes,
        "distributed_type": accelerator.distributed_type.name,
        "precision": precision,
        "dataloader": {"split_batches": False, "even_batches": True, "drop_last": False},
        "source_sha256": {name: file_sha256(source_root / name) for name in ("data.py", "sd.py", "sd_lora.py")},
        "packages": packages,
    }


def _save_adapter(config: dict, unet, directory: Path, base_lock: dict,
                  signature: str, state: dict) -> None:
    """Write Diffusers inference weights and a small raw PEFT resume state.

    The two files contain the same trained adapter in different naming schemes.
    Diffusers' format is consumed by sd.build_pipeline; the raw PEFT format can
    be restored into the existing UNet without reloading several GB of frozen
    base weights. No base UNet, text encoder, or VAE state is serialized here.
    """
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    peft_state = {key: value.detach().float().cpu().contiguous() for key, value in
                  get_peft_model_state_dict(unet, adapter_name=ADAPTER_NAME).items()}
    if not peft_state:
        raise RuntimeError("Refusing to save an empty LoRA adapter")
    raw_path = directory / "training_adapter.safetensors"
    raw_temporary = directory / "training_adapter.safetensors.part"
    save_file(peft_state, str(raw_temporary))
    raw_temporary.replace(raw_path)
    # Diffusers can infer rank from tensor shapes, but bare tensor files do not
    # carry PEFT's lora_alpha. Its default inference scaling is alpha=rank.
    # Bake our alpha/rank into the inference B matrix so any configured alpha
    # gives exactly the trained low-rank update. Keep the raw resume tensors
    # unchanged, since the training UNet already applies its configured alpha.
    inference_scale = float(config["train"]["lora_alpha"]) / config["train"]["lora_rank"]
    inference_state = {key: value * inference_scale if key.endswith(".lora_B.weight") else value
                       for key, value in peft_state.items()}
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(directory),
        unet_lora_layers=convert_state_dict_to_diffusers(inference_state),
        safe_serialization=True,
        weight_name="pytorch_lora_weights.safetensors",
    )
    # Metadata is published last. Shared inference checks these exact identity
    # fields before it accepts the adapter, and hashes the inference weight file
    # before deciding whether an existing image-generation run can be resumed.
    write_json(directory / "adapter_metadata.json", {
        "base_model_id": base_lock["repo_id"],
        "base_model_revision": base_lock["revision"],
        "dataset_revision": config["data"]["revision"],
        "train_manifest_sha256": manifest_sha(config, "train"),
        "val_manifest_sha256": manifest_sha(config, "val"),
        "encoding_version": ENCODING_VERSION,
        "text_config": config["text"],
        "global_step": state["global_step"],
        "training_signature": signature,
        "adapter_name": ADAPTER_NAME,
        "inference_alpha_convention": "alpha_over_rank_baked_into_lora_B; inference_alpha_equals_rank",
        "peft_state_keys": sorted(peft_state),
        "training_adapter_sha256": file_sha256(raw_path),
        "inference_adapter_sha256": file_sha256(directory / "pytorch_lora_weights.safetensors"),
    })


def _restore_adapter(config: dict, unet, directory: Path, signature: str) -> None:
    """Validate and restore only PEFT tensors into the freshly loaded base."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    identity = adapter_identity(config, directory)
    metadata = identity["metadata"]
    if metadata.get("training_signature") != signature or metadata.get("adapter_name") != ADAPTER_NAME:
        raise ValueError("Resume adapter has a different training signature or adapter name")
    raw_path = directory / "training_adapter.safetensors"
    if file_sha256(raw_path) != metadata.get("training_adapter_sha256"):
        raise ValueError("Resume PEFT adapter checksum mismatch")
    if identity["weights_sha256"] != metadata.get("inference_adapter_sha256"):
        raise ValueError("Resume inference adapter checksum mismatch")
    restored = load_file(str(raw_path), device="cpu")
    expected = get_peft_model_state_dict(unet, adapter_name=ADAPTER_NAME)
    if set(restored) != set(expected) or sorted(restored) != metadata.get("peft_state_keys"):
        raise ValueError("Resume adapter tensor names do not exactly match this UNet's LoRA parameters")
    if any(restored[key].shape != expected[key].shape for key in expected):
        raise ValueError("Resume adapter tensor shapes do not match the configured LoRA")
    result = set_peft_model_state_dict(unet, restored, adapter_name=ADAPTER_NAME)
    # PEFT may report all frozen base tensors as missing; those are intentionally
    # absent. Unexpected keys are never acceptable. A value comparison confirms
    # that PEFT's adapter-name remapping actually reached every expected tensor.
    if result is not None and result.unexpected_keys:
        raise ValueError(f"Unexpected PEFT resume keys: {result.unexpected_keys}")
    loaded = get_peft_model_state_dict(unet, adapter_name=ADAPTER_NAME)
    if any(not torch.equal(loaded[key].detach().float().cpu(), restored[key].float()) for key in expected):
        raise ValueError("PEFT did not restore every adapter tensor exactly")


def _find_resume(config: dict, requested: str, *, allow_missing: bool = False) -> Path | None:
    """Choose a fully published checkpoint, ignoring incomplete save folders."""
    if requested != "latest":
        directory = resolve_path(config, requested)
    else:
        root = training_root(config) / "checkpoints"
        candidates = [path for path in root.glob("step_*") if path.is_dir()
                      and path.name.removeprefix("step_").isdigit()
                      and (path / "checkpoint_complete.json").is_file()]
        if not candidates:
            if allow_missing:
                return None
            raise FileNotFoundError("No complete training checkpoint exists for --resume latest")
        directory = max(candidates, key=lambda path: int(path.name.removeprefix("step_")))
    for filename in ("trainer_state.json", "training_signature.json", "checkpoint_complete.json"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Resume requires a complete training checkpoint: missing {directory / filename}")
    return directory


def _read_resume_state(directory: Path, signature: str, epoch_batches: int,
                       max_steps: int, accumulation: int) -> dict:
    """Validate the saved cursor before Accelerate restores optimizer/RNG state."""
    state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    signature_record = json.loads((directory / "training_signature.json").read_text(encoding="utf-8"))
    complete = json.loads((directory / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if complete.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported training checkpoint version")
    if signature_record.get("sha256") != signature or json_sha256(signature_record.get("details")) != signature:
        raise ValueError("Checkpoint training signature differs from this configuration, code, models, data, or world size")
    expected_rng_names = {_strict_rng_name(index) for index in range(signature_record["details"]["world_size"])}
    rng_hashes = complete.get("strict_rng_sha256", {})
    if set(rng_hashes) != expected_rng_names:
        raise ValueError("The complete checkpoint lacks strict RNG states for every training process")
    for filename, expected_hash in rng_hashes.items():
        if file_sha256(directory / filename) != expected_hash:
            raise ValueError(f"Checkpoint RNG state changed or is incomplete: {filename}")
    if state.get("training_signature") != signature or complete.get("training_signature") != signature:
        raise ValueError("Checkpoint completion/state signatures disagree")
    for key in ("epoch", "next_batch", "global_step", "microbatches_seen"):
        if isinstance(state.get(key), bool) or not isinstance(state.get(key), int) or state[key] < 0:
            raise ValueError(f"Invalid checkpoint cursor field: {key}")
    if state["next_batch"] >= epoch_batches or state["next_batch"] % accumulation:
        raise ValueError("Resume cursor is not at an optimizer boundary in this prepared DataLoader")
    if state["microbatches_seen"] != state["epoch"] * epoch_batches + state["next_batch"]:
        raise ValueError("Checkpoint epoch/batch cursor disagrees with its consumed-microbatch count")
    if not 0 < state["global_step"] <= max_steps or complete.get("global_step") != state["global_step"]:
        raise ValueError("Invalid or inconsistent completed optimizer step")
    best = state.get("best_score")
    best_step = state.get("best_step")
    if best is not None and (isinstance(best, bool) or not isinstance(best, (int, float)) or not math.isfinite(best)):
        raise ValueError("Invalid checkpoint best validation CLIPScore")
    if (best is None and best_step is not None) or (best is not None and
            (isinstance(best_step, bool) or not isinstance(best_step, int) or not 0 < best_step <= state["global_step"])):
        raise ValueError("Invalid checkpoint best validation step")
    validated = state.get("last_validation_step")
    if validated is not None and (isinstance(validated, bool) or not isinstance(validated, int)
                                  or not 0 < validated <= state["global_step"]):
        raise ValueError("Invalid checkpoint last validation step")
    if best is not None and (validated is None or best_step > validated):
        raise ValueError("Checkpoint best step lies after its most recent validation")
    return state


def _quarantine_resume_directory(root: Path, area: str, source: Path) -> dict:
    """Rename one abandoned directory within its verified training subfolder.

    Quarantine preserves every file for inspection; no recursive delete is
    performed. Resolving both endpoints before the rename also prevents a
    symlink or Windows junction from turning this operation into a move outside
    the intended validation/checkpoints directory.
    """
    root = root.resolve()
    if area not in {"validation", "checkpoints"}:
        raise ValueError("Unexpected resume quarantine area")
    parent = root / area
    if parent.resolve() != parent or source.parent.resolve() != parent or source.resolve() != parent / source.name:
        raise ValueError(f"Resume quarantine path escapes its intended training subfolder: {source}")
    if not source.is_dir():
        raise ValueError(f"Resume quarantine expects a directory: {source}")
    destination = parent / f".{source.name}.abandoned_{uuid.uuid4().hex}"
    if destination.resolve().parent != parent or destination.exists():
        raise ValueError(f"Invalid resume quarantine destination: {destination}")
    source.rename(destination)
    return {"original": str(source.relative_to(root)), "quarantined": str(destination.relative_to(root))}


def _recover_saved_best(config: dict, signature: str, state: dict) -> None:
    """Restore the best that belongs to the saved optimizer-state timeline.

    A crash can occur after validation publishes a new best but before the next
    optimizer checkpoint completes. The resumed checkpoint must recover its own
    earlier best, otherwise a weight update from the abandoned future would
    influence model selection. Retained validation adapter snapshots provide
    the required historical weights without storing a second frozen base.
    Validation directories newer than the saved validation cursor belong to
    the abandoned timeline and are quarantined before their step is replayed.
    This also prevents their generation signatures from blocking regenerated
    outputs when a replay's floating-point weights differ from the first try.
    Keep the complete experiment folder when moving a run to another machine.
    """
    root = training_root(config).resolve()
    validation_root = root / "validation"
    destination = root / "checkpoints" / "best"
    quarantined = []
    try:
        if validation_root.exists():
            if validation_root.resolve() != validation_root:
                raise ValueError("The validation directory must remain within the resolved training tree")
            saved_validation_step = state["last_validation_step"]
            for path in sorted(validation_root.glob("step_*")):
                suffix = path.name.removeprefix("step_")
                if path.is_dir() and suffix.isdigit() and (saved_validation_step is None or int(suffix) > saved_validation_step):
                    quarantined.append(_quarantine_resume_directory(root, "validation", path))
        if state["best_score"] is None and (destination.exists() or destination.is_symlink()):
            # An older optimizer checkpoint may predate the first validation.
            # Its active timeline has no selected best, even if the interrupted
            # future published one. Keep that artifact under a quarantine name.
            quarantined.append(_quarantine_resume_directory(root, "checkpoints", destination))
    finally:
        if quarantined:
            write_json(root / "resume_recovery" / f"step_{state['global_step']:08d}_{uuid.uuid4().hex}.json", {
                "training_signature": signature, "restored_global_step": state["global_step"],
                "saved_last_validation_step": state["last_validation_step"],
                "saved_best_step": state["best_step"], "quarantined": quarantined,
            })
    if state["best_score"] is None:
        return
    snapshot = validation_root / f"step_{state['best_step']:08d}" / "adapter"
    identity = adapter_identity(config, snapshot)
    metadata = identity["metadata"]
    if metadata.get("global_step") != state["best_step"] or metadata.get("training_signature") != signature:
        raise ValueError("The historical best adapter snapshot does not match the saved training state")
    if identity["weights_sha256"] != metadata.get("inference_adapter_sha256"):
        raise ValueError("Historical best inference adapter checksum mismatch")
    if file_sha256(snapshot / "training_adapter.safetensors") != metadata.get("training_adapter_sha256"):
        raise ValueError("Historical best raw PEFT adapter checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("training_adapter.safetensors", "pytorch_lora_weights.safetensors"):
        temporary = destination / f"{filename}.part"
        shutil.copyfile(snapshot / filename, temporary)
        temporary.replace(destination / filename)
    write_json(destination / "adapter_metadata.json", metadata)


def _install_adapter_state_hooks(accelerator, config: dict, unet, base_lock: dict,
                                 signature: str, state: dict) -> None:
    """Replace Accelerate's full-model serialization with adapter serialization.

    Default optimizer, registered scheduler, AMP scaler, and per-rank RNG saving
    remain enabled. Removing the model/weight entries from these prehooks is
    essential: merely writing a LoRA file alongside save_state would still
    serialize the entire frozen SDXL UNet.
    """
    def save_hook(models, weights, output_dir):
        if len(models) != 1 or accelerator.unwrap_model(models[0]) is not accelerator.unwrap_model(unet):
            raise RuntimeError("This checkpoint hook supports exactly one prepared SDXL UNet")
        _main_process_call(
            accelerator,
            lambda: _save_adapter(config, accelerator.unwrap_model(unet), Path(output_dir), base_lock, signature, state),
            "Saving adapter tensors",
        )
        while weights:
            weights.pop()

    def load_hook(models, input_dir):
        if len(models) != 1 or accelerator.unwrap_model(models[0]) is not accelerator.unwrap_model(unet):
            raise RuntimeError("This resume hook supports exactly one prepared SDXL UNet")
        _restore_adapter(config, accelerator.unwrap_model(unet), Path(input_dir), signature)
        while models:
            models.pop()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def _save_training_checkpoint(accelerator, config: dict, signature_record: dict, state: dict) -> Path:
    """Publish a checkpoint only after every rank has saved its own RNG state."""
    root = training_root(config) / "checkpoints"
    target = root / f"step_{state['global_step']:08d}"
    # A fresh staging name lets --resume latest ignore a directory left by a
    # crash during saving. Existing completed checkpoints are never replaced.
    staging_name = _main_process_call(accelerator, lambda: f".step_{state['global_step']:08d}.{uuid.uuid4().hex}.incomplete",
                                      "Choosing a checkpoint staging directory")
    temporary = root / staging_name
    accelerator.wait_for_everyone()

    def make_directory():
        if target.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint destination already exists: {target} or {temporary}; inspect it before retrying")
        temporary.mkdir(parents=True)

    _main_process_call(accelerator, make_directory, "Creating checkpoint directory")
    accelerator.save_state(str(temporary))
    _save_strict_rng(accelerator, temporary)
    accelerator.wait_for_everyone()

    def publish():
        write_json(temporary / "trainer_state.json", state)
        write_json(temporary / "training_signature.json", signature_record)
        write_json(temporary / "checkpoint_complete.json", {
            "version": CHECKPOINT_VERSION,
            "global_step": state["global_step"],
            "training_signature": signature_record["sha256"],
            "strict_rng_sha256": {
                _strict_rng_name(index): file_sha256(temporary / _strict_rng_name(index))
                for index in range(accelerator.num_processes)
            },
        })
        temporary.rename(target)
        write_json(root / "latest.json", {"path": target.name, "global_step": state["global_step"]})
        return str(target)

    _main_process_call(accelerator, publish, "Publishing training checkpoint")
    accelerator.wait_for_everyone()
    return target


def _validate(accelerator, config: dict, components: dict, unet, records: list[dict],
               base_lock: dict, signature: str, state: dict) -> dict:
    """Select checkpoints with a fixed validation population and CLIPScore only.

    DDP-wrapped UNet forward calls cannot run on rank zero alone because they
    may synchronize buffers. The validation pipeline instead receives the
    unwrapped current UNet while other ranks wait. Its supplied adapter path is
    a provenance snapshot; it must not be loaded again on top of the live LoRA.
    """
    import torch

    accelerator.wait_for_everyone()
    folder = training_root(config) / "validation" / f"step_{state['global_step']:08d}"
    current_unet = accelerator.unwrap_model(unet)

    def run_validation():
        adapter = folder / "adapter"
        _save_adapter(config, current_unet, adapter, base_lock, signature, state)
        pipe = None
        try:
            validation_components = dict(components)
            validation_components["unet"] = current_unet
            pipe = build_pipeline(config, device=accelerator.device, components=validation_components)
            # Accelerate's mixed-precision wrapper returns fp32 UNet outputs, so
            # validation latents are fp32. SDXL upcasts an fp16 VAE for decoding
            # but not a bf16 one; decode in fp32 here, as the fp16 path already does.
            pipe.vae.to(dtype=torch.float32)
            pipe.set_adapters([ADAPTER_NAME], adapter_weights=[float(config["generation"]["lora_scale"])])
            rows = generate_images(
                config, "sd_lora", split="val", adapter=adapter, records=records,
                seed=int(config["train"]["validation_seed"]), output_dir=folder, pipeline=pipe,
            )
            images = [folder / row["image"] for row in rows]
            score = compute_clipscore(config, images, records, folder / "clipscore_per_sample.csv")
            value = float(score["clipscore"])
            if not math.isfinite(value):
                raise FloatingPointError("Validation CLIPScore is non-finite")
            result = {
                "global_step": state["global_step"], "split": "val", "samples": len(records),
                "seed": config["train"]["validation_seed"], "clipscore": value,
                "clip_details": score, "selection_metric": "maximize_validation_clipscore",
                "sample_ids": [record["sample_id"] for record in records],
                "training_signature": signature,
            }
            write_json(folder / "validation_metrics.json", result)
            if state["best_score"] is None or value > state["best_score"]:
                # Strict improvement preserves the earlier checkpoint on ties.
                _save_adapter(config, current_unet, training_root(config) / "checkpoints" / "best",
                              base_lock, signature, state)
                result["improved"] = True
            else:
                result["improved"] = False
            return result
        finally:
            # build_pipeline and SDXL's VAE upcast path can change VAE dtypes.
            # Training always encodes pixels in fp32, regardless of inference
            # precision. Reset the adapter scale and training mode as well.
            current_unet.set_adapters([ADAPTER_NAME], weights=[1.0])
            components["vae"].to(device=accelerator.device, dtype=torch.float32)
            current_unet.train()
            components["text_encoder"].eval()
            components["text_encoder_2"].eval()
            components["vae"].eval()
            del pipe
            gc.collect()
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()

    with _preserve_training_rng():
        result = _main_process_call(accelerator, run_validation, "Validation")
    accelerator.wait_for_everyone()
    if result["improved"]:
        state["best_score"] = result["clipscore"]
        state["best_step"] = state["global_step"]
    state["last_validation_step"] = state["global_step"]
    accelerator.print(f"validation step={state['global_step']} CLIPScore={result['clipscore']:.6f} best={state['best_score']:.6f}")
    return result


def _minimum_prompt_chunks(config: dict, tokenizers) -> int:
    """Match inference's positive context length if a negative prompt is set.

    SDXL classifier-free guidance needs equal positive/negative sequence
    lengths. sd.prompt_embeddings therefore extends the positive caption to
    cover a longer configured negative prompt. Training applies the same rule
    even though its denoising target uses only the positive conditioning.
    """
    negative = config["generation"].get("negative_prompt", "")
    if not negative:
        return 1
    return max(1, *(math.ceil(len(tokenizer(negative, add_special_tokens=False, truncation=False)["input_ids"])
                              / (tokenizer.model_max_length - 2)) for tokenizer in tokenizers))


def train(config: dict, resume: str | None = None, *, auto_resume: bool = False) -> dict:
    """Train only UNet attention LoRA parameters with single-device or DDP AMP.

    Checkpoints are taken only at optimizer boundaries. A cursor identifies the
    next *prepared, per-rank* batch, so the same world size and epoch sampler
    reconstruct the data order on resume. The DataLoader owns a private torch
    generator; creating or skipping its iterator cannot consume the generator
    used for VAE sampling, diffusion noise, or timesteps.
    """
    _check_training_config(config)
    try:
        import torch
        import torch.nn.functional as functional
        from accelerate import Accelerator, DataLoaderConfiguration
        from diffusers import DDPMScheduler
        from diffusers.optimization import get_scheduler
        from peft import LoraConfig
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("Training dependencies are missing; install the project's pinned requirements in the intended training environment") from error

    requested_device, _, precision = runtime_device_dtype(config)
    if requested_device.type not in {"cpu", "cuda"} or requested_device.index is not None:
        raise ValueError("Training supports runtime.device=cpu or cuda; select CUDA GPUs with CUDA_VISIBLE_DEVICES/Accelerate")
    accumulation = config["train"]["gradient_accumulation_steps"]
    accelerator = Accelerator(
        cpu=requested_device.type == "cpu",
        mixed_precision="no" if precision == "fp32" else precision,
        gradient_accumulation_steps=accumulation,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(split_batches=False, even_batches=True),
    )
    if accelerator.distributed_type.name not in {"NO", "MULTI_GPU", "MULTI_CPU"}:
        raise ValueError("Only a single device or standard Accelerate DDP is supported; FSDP, DeepSpeed, and other backends are unsupported")
    device, weight_dtype, actual_precision = runtime_device_dtype(config, accelerator.device)
    if actual_precision != precision:
        raise ValueError("All training devices must support the same configured precision")

    dataset = None
    try:
        with ExitStack() as cleanup:
            root = training_root(config)
            # Rank zero owns the training lock for the entire job. Other ranks
            # receive a lock failure through the same control-plane broadcast.
            _main_process_call(
                accelerator,
                lambda: cleanup.enter_context(exclusive_file(root / ".training.lock", timeout=5.0)),
                "Acquiring the training lock",
            )
            # First-use Hub revision resolution belongs to rank zero. The
            # barrier/broadcast completes the writes before other ranks read.
            base_lock = _main_process_call(accelerator, lambda: model_lock(config), "Pinning the base model")
            clip_lock = _main_process_call(accelerator, lambda: model_lock(config, "clip"), "Pinning the CLIP evaluator")
            accelerator.wait_for_everyone()
            if model_lock(config) != base_lock or model_lock(config, "clip") != clip_lock:
                raise ValueError("Ranks read inconsistent model locks")

            train_records = load_split(config, "train")
            val_records = load_split(config, "val")
            if not train_records or not val_records:
                raise ValueError("Prepared train and validation splits must both be nonempty; run data.py prepare first")
            count = min(config["train"]["validation_samples"], len(val_records))
            validation_records = sorted(
                random.Random(config["train"]["validation_seed"]).sample(val_records, count),
                key=lambda record: record["sample_id"],
            )
            details = _training_signature(config, base_lock, clip_lock, accelerator, precision, validation_records)
            signature_record = {"sha256": json_sha256(details), "details": details}
            signature = signature_record["sha256"]
            def find_resume_directory():
                if resume is None and not auto_resume:
                    return None
                requested = resume if resume is not None else "latest"
                directory = _find_resume(config, requested, allow_missing=auto_resume)
                return str(directory) if directory is not None else None

            # Select the checkpoint under the same lock used to publish it.
            resume_directory = _main_process_call(accelerator, find_resume_directory, "Finding the resume checkpoint")

            def initialize_artifacts():
                marker = root / "training_signature.json"
                if marker.exists():
                    saved = json.loads(marker.read_text(encoding="utf-8"))
                    if saved != signature_record:
                        raise ValueError("This output_dir already contains incompatible training settings; use a new output_dir")
                    if resume_directory is None:
                        if not auto_resume:
                            raise FileExistsError("Training artifacts already exist; use --resume latest or choose a new experiment.output_dir")
                        # A failed download or model setup can leave a signature
                        # before any update has been recorded. Retry that setup
                        # in place, but preserve recorded work if no checkpoint
                        # is available to restore the optimizer and RNGs.
                        if (root / "training_log.jsonl").exists() or (root / "checkpoints").exists():
                            raise FileExistsError("Training artifacts exist without a complete checkpoint; choose a new experiment.output_dir to restart")
                write_json(marker, signature_record)
                write_json(root / "validation_selection.json", {
                    "split": "val", "seed": config["train"]["validation_seed"],
                    "sample_ids": [record["sample_id"] for record in validation_records],
                    "val_manifest_sha256": details["val_manifest_sha256"],
                    "selection_metric": "maximize_validation_clipscore", "ties": "retain_earlier_step",
                })

            _main_process_call(accelerator, initialize_artifacts, "Initializing training provenance")
            # Identical initialization on every rank is required by DDP. LoRA
            # starts from PEFT's Gaussian initialization with a zero output
            # branch. Rank-specific training RNGs are assigned after setup.
            seed = config["experiment"]["train_seed"]
            set_seed(seed)
            pipe = build_pipeline(config, device=device)
            components = dict(pipe.components)
            unet, vae = pipe.unet, pipe.vae
            tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
            encoders = [pipe.text_encoder, pipe.text_encoder_2]
            for module in (unet, vae, *encoders):
                module.requires_grad_(False)
            vae.to(device=device, dtype=torch.float32)
            vae.eval()
            for encoder in encoders:
                encoder.eval()
            unet.add_adapter(LoraConfig(
                r=config["train"]["lora_rank"], lora_alpha=config["train"]["lora_alpha"],
                target_modules=config["train"]["target_modules"], init_lora_weights="gaussian",
            ), adapter_name=ADAPTER_NAME)
            trainable = []
            for name, parameter in unet.named_parameters():
                if parameter.requires_grad:
                    if "lora_" not in name:
                        raise RuntimeError(f"A non-LoRA UNet parameter was unexpectedly made trainable: {name}")
                    # Optimizer/master adapter parameters stay fp32 even when
                    # the frozen base and autocast computations use fp16/bf16.
                    parameter.data = parameter.data.float()
                    trainable.append(parameter)
            if not trainable:
                raise RuntimeError("No LoRA parameters matched train.target_modules")
            if config["train"]["gradient_checkpointing"]:
                unet.enable_gradient_checkpointing()
            unet.train()
            # Use the original checkpoint's training noise schedule, not the
            # shared inference pipeline's DPMSolver sampling schedule.
            noise_scheduler = DDPMScheduler.from_pretrained(
                base_lock["repo_id"], revision=base_lock["revision"], subfolder="scheduler",
                cache_dir=str(resolve_path(config, config["paths"]["cache"]) / "huggingface"),
                local_files_only=config["runtime"].get("local_files_only", False),
            )
            if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
                raise ValueError("Training supports DDPM epsilon or v_prediction targets only")
            minimum_chunks = _minimum_prompt_chunks(config, tokenizers)
            del pipe

            settings = config["train"]
            optimizer = torch.optim.AdamW(
                trainable, lr=settings["learning_rate"],
                betas=(settings["adam_beta1"], settings["adam_beta2"]),
                weight_decay=settings["adam_weight_decay"], eps=settings["adam_epsilon"],
            )
            # The scheduler refers to the underlying optimizer. It is registered
            # explicitly and advanced manually once per *successful* optimizer
            # update, so DDP world size and fp16 overflow skips cannot multiply
            # the learning-rate schedule's speed.
            lr_scheduler = get_scheduler(
                settings["lr_scheduler"], optimizer=optimizer,
                num_warmup_steps=settings["warmup_steps"], num_training_steps=settings["max_train_steps"],
            )
            dataset = FoodDataset(config, train_records, settings["resolution"])
            sampler = EpochSampler(len(dataset), seed)
            loader_generator = torch.Generator(device="cpu").manual_seed(seed)
            loader = DataLoader(
                dataset, batch_size=1, sampler=sampler, shuffle=False, drop_last=False,
                collate_fn=collate_training_batch, num_workers=settings["num_workers"],
                pin_memory=device.type == "cuda", persistent_workers=False, generator=loader_generator,
            )
            unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
            accelerator.register_for_checkpointing(lr_scheduler)
            epoch_batches = len(loader)
            if epoch_batches < 1:
                raise ValueError("The prepared per-rank training DataLoader is empty")
            state = {
                "training_signature": signature, "epoch": 0, "next_batch": 0,
                "global_step": 0, "microbatches_seen": 0,
                "best_score": None, "best_step": None, "last_validation_step": None,
            }
            _install_adapter_state_hooks(accelerator, config, unet, base_lock, signature, state)
            # DDP ranks need distinct diffusion noise while retaining identical
            # adapter initialization and a common deterministic sample order.
            # On resume, load_state replaces these seeds with saved exact RNGs.
            set_seed(seed + accelerator.process_index)
            if resume_directory is not None:
                restored_state = _read_resume_state(Path(resume_directory), signature, epoch_batches,
                                                    settings["max_train_steps"], accumulation)
                state.update(restored_state)
                accelerator.load_state(resume_directory)
                _restore_strict_rng(accelerator, Path(resume_directory))
                # This runs for every resume, including checkpoints that have
                # not validated yet and therefore have no saved best adapter.
                _main_process_call(accelerator, lambda: _recover_saved_best(config, signature, state),
                                   "Recovering the saved best adapter")
                accelerator.print(f"Resumed {resume_directory}: step={state['global_step']} epoch={state['epoch']} next_batch={state['next_batch']}")

            nominal_batch = accumulation * accelerator.num_processes
            actual = {
                "training_signature": signature, "world_size": accelerator.num_processes,
                "batch_size_per_gpu": 1, "gradient_accumulation_steps": accumulation,
                "nominal_effective_batch_size": nominal_batch,
                "prepared_batches_per_rank_per_epoch": epoch_batches,
                "final_accumulation_group_batches": epoch_batches % accumulation or accumulation,
                "final_accumulation_group_effective_batch_size": (epoch_batches % accumulation or accumulation) * accelerator.num_processes,
                "ddp_padding_note": "even_batches repeats initial training samples when needed to give all ranks equal batch counts",
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "frozen_vae_dtype": "float32", "frozen_weight_dtype": str(weight_dtype),
                "trainable_dtype": "float32", "precision": precision,
                "prediction_type": noise_scheduler.config.prediction_type,
                "resume_checkpoint": resume_directory,
            }
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _main_process_call(accelerator, lambda: save_run_metadata(config, root / "environments" / f"{stamp}.json", actual),
                               "Saving training environment")
            accelerator.print(f"Training LoRA: {actual['trainable_parameters']:,} parameters; nominal effective batch={nominal_batch}")
            optimizer.zero_grad(set_to_none=True)
            last_checkpoint_step = state["global_step"] if resume_directory is not None else None

            while state["global_step"] < settings["max_train_steps"]:
                epoch, offset = state["epoch"], state["next_batch"]
                sampler.set_epoch(epoch)
                if hasattr(loader, "set_epoch"):
                    loader.set_epoch(epoch)
                loader_generator.manual_seed(seed + epoch)
                epoch_loader = accelerator.skip_first_batches(loader, offset) if offset else loader
                # skip_first_batches constructs a new DataLoaderShard whose
                # internal iteration counter starts at zero. Set the epoch on
                # that returned loader as well; setting only the original
                # loader would replay the epoch-zero permutation on resume.
                if hasattr(epoch_loader, "set_epoch"):
                    epoch_loader.set_epoch(epoch)
                sampler.set_epoch(epoch)
                accumulated_loss = 0.0
                for batch_index, batch in enumerate(epoch_loader, start=offset):
                    # Accelerate flushes accumulated gradients at the end of a
                    # prepared DataLoader. Correct its fixed 1/G loss scaling
                    # for a shorter final group, giving an average over the
                    # actual M microbatches instead of an underweighted M/G.
                    group_start = (batch_index // accumulation) * accumulation
                    group_size = min(accumulation, epoch_batches - group_start)
                    with accelerator.accumulate(unet):
                        with torch.no_grad():
                            # SDXL's VAE can overflow in fp16. Both its weights
                            # and the complete encoder computation remain fp32.
                            with torch.autocast(device_type=device.type, enabled=False):
                                pixels = batch["pixel_values"].to(device=device, dtype=torch.float32)
                                latents = vae.encode(pixels).latent_dist.sample()
                                latents = latents * vae.config.scaling_factor
                            latents = latents.to(dtype=weight_dtype)
                            prompt, pooled, _ = encode_caption(
                                config, tokenizers, encoders, batch["captions"][0],
                                device, weight_dtype, minimum_chunks=minimum_chunks,
                            )
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                                  (latents.shape[0],), device=device, dtype=torch.long)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)
                        target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                        with accelerator.autocast():
                            prediction = unet(
                                noisy_latents, timesteps, encoder_hidden_states=prompt,
                                added_cond_kwargs={"text_embeds": pooled,
                                                   "time_ids": batch["time_ids"].to(device=device, dtype=weight_dtype)},
                                return_dict=False,
                            )[0]
                            loss = functional.mse_loss(prediction.float(), target.float(), reduction="mean")
                        finite_ranks = accelerator.reduce(torch.isfinite(loss.detach()).to(dtype=torch.int32), reduction="sum")
                        if finite_ranks.item() != accelerator.num_processes:
                            raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}, sample={batch['sample_ids']}")
                        mean_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                        accumulated_loss += mean_loss / group_size
                        accelerator.backward(loss * (accumulation / group_size))
                        if accelerator.sync_gradients:
                            norm = accelerator.clip_grad_norm_(trainable, settings["max_grad_norm"])
                            if precision != "fp16" and not torch.isfinite(norm).all().item():
                                raise FloatingPointError("Non-finite LoRA gradient norm")
                        optimizer.step()
                        did_update = accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
                        if did_update:
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    state["microbatches_seen"] += 1
                    if batch_index + 1 == epoch_batches:
                        state["epoch"], state["next_batch"] = epoch + 1, 0
                    else:
                        state["epoch"], state["next_batch"] = epoch, batch_index + 1
                    # Release training activation references before the
                    # memory-intensive, no-grad validation pipeline is created.
                    del pixels, latents, prompt, pooled, noise, timesteps, noisy_latents, target, prediction, loss
                    if not accelerator.sync_gradients:
                        continue
                    if not did_update:
                        accelerator.print(f"Skipped fp16 optimizer update at epoch={epoch}, batch={batch_index}; scheduler/global_step unchanged")
                        accumulated_loss = 0.0
                        continue
                    state["global_step"] += 1
                    step = state["global_step"]
                    log = {
                        "global_step": step, "epoch": epoch, "next_batch": state["next_batch"],
                        "loss": accumulated_loss, "learning_rate": lr_scheduler.get_last_lr()[0],
                        "effective_batch_size": group_size * accelerator.num_processes,
                    }

                    def write_log():
                        with (root / "training_log.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(json.dumps(log, allow_nan=False) + "\n")

                    _main_process_call(accelerator, write_log, "Writing training progress")
                    accelerator.print(f"step={step}/{settings['max_train_steps']} loss={accumulated_loss:.6f} lr={log['learning_rate']:.3g}")
                    accumulated_loss = 0.0
                    if step % settings["validate_every_steps"] == 0 or step == settings["max_train_steps"]:
                        _validate(accelerator, config, components, unet, validation_records, base_lock, signature, state)
                    if step % settings["checkpoint_every_steps"] == 0 or step == settings["max_train_steps"]:
                        _save_training_checkpoint(accelerator, config, signature_record, state)
                        last_checkpoint_step = step
                    if step >= settings["max_train_steps"]:
                        break

            # A final adapter is always available independently of whether it
            # improved validation. The final training state is also resumable.
            if state["last_validation_step"] != state["global_step"]:
                _validate(accelerator, config, components, unet, validation_records, base_lock, signature, state)
            if last_checkpoint_step != state["global_step"]:
                _save_training_checkpoint(accelerator, config, signature_record, state)
            accelerator.wait_for_everyone()
            _main_process_call(
                accelerator,
                lambda: _save_adapter(config, accelerator.unwrap_model(unet), root / "checkpoints" / "last", base_lock, signature, state),
                "Saving the final adapter",
            )
            _main_process_call(accelerator, lambda: write_json(root / "training_summary.json", {
                **state, **actual,
                "best_adapter": str(root / "checkpoints" / "best"),
                "last_adapter": str(root / "checkpoints" / "last"),
                "checkpoint_selection": "fixed validation CLIPScore only; strict improvement; test examples never select weights",
            }), "Saving training summary")
            accelerator.wait_for_everyone()
            return state
    finally:
        if dataset is not None:
            dataset.store.close()
        accelerator.end_training()


def run_from_ide(config: dict) -> None:
    """Run training and test evaluation in one single-device IDE process."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("For distributed training, launch sd_lora.py with the explicit train subcommand.")
    root = training_root(config)
    adapter = root / "checkpoints" / "best"
    if (root / "training_summary.json").is_file():
        # The summary is published only after training and validation finish.
        # Check the saved settings so a changed config does not silently use an
        # adapter from a previous experiment. Inference validates the adapter
        # metadata, base-model revision, and prepared data before generating.
        saved = json.loads((root / "training_signature.json").read_text(encoding="utf-8"))
        if saved["details"]["config"] != {key: value for key, value in config.items() if not key.startswith("_")}:
            raise ValueError("Training settings changed; use a new experiment.output_dir in config.yaml.")
        print(f"Using the completed experiment's best LoRA adapter at {adapter}", flush=True)
    else:
        # A best adapter can exist before the run is finished. Resume the newest
        # published optimizer checkpoint in that case, rather than evaluating
        # a partially trained model or restarting from the pretrained weights.
        print("Preparing LoRA training; looking for a complete checkpoint...", flush=True)
        train(config, auto_resume=True)

    print("Running SDXL + LoRA on the full test split...", flush=True)
    inference_commands(config, "sd_lora", "all", "test", adapter=adapter)
    report_when_ready(config)


def main() -> None:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    if len(sys.argv) == 1:
        run_from_ide(load_config(default_config))
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "generate", "evaluate", "all"))
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--adapter", type=Path, help="Inference adapter directory; default: outputs/.../sd_lora/checkpoints/best")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="CHECKPOINT",
                        help="For train: resume latest complete checkpoint, or provide its directory")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, help="Inference-only diagnostic population; omit for full final evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "train":
        if args.adapter is not None or args.limit is not None:
            parser.error("train uses --resume for training state; --adapter and --limit are inference-only")
        train(config, args.resume)
    else:
        if args.resume is not None:
            parser.error("--resume applies only to train; select inference weights with --adapter")
        adapter = resolve_path(config, args.adapter) if args.adapter is not None else training_root(config) / "checkpoints" / "best"
        if not adapter.is_dir():
            parser.error(f"Adapter directory does not exist: {adapter}; finish training or provide --adapter")
        # Shared dispatch makes all exactly generate + evaluate. No inference
        # command can enter the training function or silently create weights.
        inference_commands(config, "sd_lora", args.command, args.split, adapter=adapter, limit=args.limit)


if __name__ == "__main__":
    main()
