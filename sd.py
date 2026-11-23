"""Original SDXL baseline and the functions shared by both experiment groups.

The baseline does not train any weights. This module owns the *single* image
generation and scoring implementation used by both sd.py and sd_lora.py.
Running this file with no arguments generates and scores the full test set.
Importing it does not load models, download weights, or launch GPU work.

Authoring status: this source was written without executing it, as requested.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from data import (
    ParquetImageStore,
    file_sha256,
    json_sha256,
    load_config,
    load_split,
    manifest_sha,
    read_jsonl,
    resolve_path,
    write_json,
    write_jsonl,
)


ENCODING_VERSION = "sdxl-full-caption-chunks-v1"
GENERATION_VERSION = "food500cap-generation-v1"


def run_root(config: dict) -> Path:
    """Keep checkpoints, model locks, and both groups under one experiment."""
    return resolve_path(config, config["experiment"]["output_dir"])


@contextmanager
def exclusive_file(path: Path, timeout: float = 120.0):
    """Serialize small manifest/cache updates made by independent cluster jobs.

    The file is a coordination marker, not an operating-system advisory lock.
    A crashed process can leave a marker behind; after checking that no job is
    using it, the user can remove that marker and retry. We never silently
    remove an existing marker, because another job may still own it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Another process holds {path}; check for a stale lock.")
            time.sleep(0.25)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def model_lock(config: dict, kind: str = "base") -> dict:
    """Resolve a Hub revision once, then require both groups to reuse its SHA.

    A floating name such as 'main' is convenient for initial setup but is not
    a reproducible model identity. The first caller writes an immutable lock.
    Later callers read the saved SHA even if the Hub branch subsequently moves.
    Offline first-use requires an explicit commit SHA in config.yaml.
    """
    if kind not in {"base", "clip"}:
        raise ValueError(f"Unknown model-lock kind: {kind}")
    if kind == "base":
        repository = config["model"]["id"]
        requested = config["model"].get("revision", "main")
    else:
        repository = config["evaluation"]["clip_model"]
        requested = config["evaluation"].get("clip_revision", "main")
    target = run_root(config) / f"{kind}_model_lock.json"
    with exclusive_file(target.with_suffix(".lock")):
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved["repo_id"] != repository or requested not in {
                saved["requested_revision"], saved["revision"]
            }:
                raise ValueError("Model configuration conflicts with the experiment lock; use a new output_dir.")
            return saved
        if len(requested) == 40 and all(character in "0123456789abcdef" for character in requested.lower()):
            revision = requested.lower()
        elif config["runtime"].get("local_files_only", False):
            raise ValueError("Offline first-use needs an explicit model commit SHA or an existing model lock.")
        else:
            from huggingface_hub import HfApi

            revision = HfApi().model_info(repository, revision=requested, token=False).sha
        saved = {"repo_id": repository, "requested_revision": requested, "revision": revision}
        write_json(target, saved)
        return saved


def runtime_device_dtype(config: dict, device=None):
    """Choose one device/precision policy for training and inference."""
    import torch

    selected = torch.device(device or config["runtime"].get("device", "cuda"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; configure the intended GPU environment first.")
    precision = config["runtime"].get("precision", "auto")
    if precision == "auto":
        precision = "bf16" if selected.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if selected.type == "cuda" else "fp32"
        )
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be auto, fp32, fp16, or bf16")
    if selected.type != "cuda" and precision != "fp32":
        raise ValueError("The reference CPU path uses fp32; mixed precision is intended for CUDA.")
    if precision == "bf16" and selected.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 is not supported by the selected CUDA device.")
    torch.backends.cuda.matmul.allow_tf32 = bool(config["runtime"].get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(config["runtime"].get("allow_tf32", False))
    torch.backends.cudnn.benchmark = False
    return selected, {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision], precision


def set_seed(seed: int) -> None:
    """Set process RNGs; per-image generators are still required for sampling."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_metadata(config: dict, target: Path, extra: dict | None = None) -> None:
    """Record the actual environment without claiming cross-platform bit equality."""
    import torch

    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "peft", "accelerate", "clean-fid", "Pillow", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_root = Path(__file__).resolve().parent
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "source_sha256": {name: file_sha256(source_root / name) for name in ("data.py", "sd.py", "sd_lora.py")},
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "extra": extra or {},
    }
    write_json(target, metadata)


def adapter_identity(config: dict, adapter: str | Path | None) -> dict | None:
    """Refuse adapters trained on a different base model or data split.

    The inference weight hash is included in generation signatures. Replacing
    the contents of a checkpoint directory therefore cannot accidentally reuse
    images previously generated from different adapter weights.
    """
    if adapter is None:
        return None
    folder = resolve_path(config, str(adapter))
    metadata_path = folder / "adapter_metadata.json"
    weights = folder / "pytorch_lora_weights.safetensors"
    if not metadata_path.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Adapter metadata/weights not found in {folder}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lock = model_lock(config)
    expected = {
        "base_model_id": lock["repo_id"], "base_model_revision": lock["revision"],
        "dataset_revision": config["data"]["revision"],
        "train_manifest_sha256": manifest_sha(config, "train"),
        "val_manifest_sha256": manifest_sha(config, "val"),
        "encoding_version": ENCODING_VERSION,
        "text_config": config["text"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Adapter provenance mismatch for {key}: {folder}")
    return {"path": str(folder), "weights_sha256": file_sha256(weights), "metadata": metadata}


def build_pipeline(config: dict, adapter=None, device=None, components: dict | None = None):
    """Create a fresh SDXL pipeline, with exactly one optional UNet adapter.

    components is used only during validation to reuse the frozen encoders,
    VAE, and current training UNet. Normal baseline and LoRA inference each
    create their own pipeline so adapter state cannot leak into the baseline.
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    selected, dtype, _ = runtime_device_dtype(config, device)
    lock = model_lock(config)
    if components is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            lock["repo_id"], revision=lock["revision"], torch_dtype=dtype,
            use_safetensors=True, add_watermarker=False,
            cache_dir=str(resolve_path(config, config["paths"]["cache"]) / "huggingface"),
            local_files_only=config["runtime"].get("local_files_only", False),
        )
    else:
        pipe = StableDiffusionXLPipeline(**components, add_watermarker=False)
    generation = config["generation"]
    if generation["scheduler"] != "DPMSolverMultistepScheduler":
        raise ValueError("This implementation fixes the shared sampler to DPMSolverMultistepScheduler.")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type=generation["algorithm_type"],
        solver_order=generation["solver_order"], use_karras_sigmas=generation["use_karras_sigmas"],
    )
    if adapter is not None:
        identity = adapter_identity(config, adapter)
        pipe.load_lora_weights(identity["path"], adapter_name="food500cap")
        pipe.set_adapters(["food500cap"], adapter_weights=[float(generation["lora_scale"])])
    # SDXL's inference implementation upcasts a fp16 VAE itself when necessary.
    # A training VAE reused for validation must first match the inference dtype;
    # the training caller restores float32 after validation finishes.
    pipe.vae.to(dtype=dtype)
    pipe.to(selected)
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    pipe.vae.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_caption(config: dict, tokenizers, text_encoders, caption: str, device, dtype, minimum_chunks: int = 1):
    """Encode the entire caption through SDXL's two frozen CLIP encoders.

    CLIP has a fixed positional window, usually 77 positions including BOS and
    EOS. We tokenize WITHOUT truncation, divide content tokens into windows,
    encode each window independently, and concatenate penultimate hidden states
    along the sequence dimension. SDXL cross-attention accepts that longer
    sequence. The second encoder's projected pooled outputs are averaged with
    content-token weights; a short caption is exactly a single-window case.

    This explicit long-prompt convention is shared by training and both image
    generators. It is not a claim that native CLIP can attend across its window.
    We process one caption per device at a time to avoid batch-dependent extra
    windows. max_chunks is a hard guard: excess text raises an error, never a
    silent truncation. Original caption strings remain unchanged in manifests.
    """
    import torch

    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("A conditioning caption must be a non-empty string.")
    if config["text"].get("long_caption_mode") != "chunk":
        raise ValueError("The full-caption implementation requires text.long_caption_mode=chunk.")
    tokens = [tokenizer(caption, add_special_tokens=False, truncation=False)["input_ids"] for tokenizer in tokenizers]
    windows = [int(tokenizer.model_max_length) for tokenizer in tokenizers]
    if any(window < 3 or window > 512 for window in windows) or len(set(windows)) != 1:
        raise ValueError("Expected matching finite CLIP context windows for SDXL's two tokenizers.")
    content_size = windows[0] - 2
    chunks = max(minimum_chunks, *(max(1, math.ceil(len(ids) / content_size)) for ids in tokens))
    if chunks > int(config["text"]["max_chunks"]):
        raise ValueError(f"Caption needs {chunks} chunks, exceeding text.max_chunks; no text was discarded.")
    hidden_parts = []
    pooled = None
    with torch.no_grad():
        for encoder_index, (tokenizer, encoder, ids) in enumerate(zip(tokenizers, text_encoders, tokens)):
            input_rows, weights = [], []
            for chunk_index in range(chunks):
                content = ids[chunk_index * content_size:(chunk_index + 1) * content_size]
                row = [tokenizer.bos_token_id, *content, tokenizer.eos_token_id]
                row += [tokenizer.pad_token_id] * (windows[0] - len(row))
                input_rows.append(row)
                weights.append(len(content))
            input_ids = torch.tensor(input_rows, device=device, dtype=torch.long)
            encoded = encoder(input_ids, output_hidden_states=True, return_dict=True)
            hidden_parts.append(encoded.hidden_states[-2].reshape(1, chunks * windows[0], -1))
            if encoder_index == 1:
                # Empty padding windows do not contribute to the pooled summary.
                content_weights = torch.tensor(weights, device=device, dtype=encoded.text_embeds.dtype)
                if content_weights.sum().item() == 0:
                    content_weights[0] = 1
                pooled = (encoded.text_embeds * content_weights[:, None]).sum(dim=0, keepdim=True) / content_weights.sum()
        embeddings = torch.cat(hidden_parts, dim=-1).to(device=device, dtype=dtype)
    return embeddings, pooled.to(device=device, dtype=dtype), {
        "token_counts_without_special": [len(ids) for ids in tokens], "chunks": chunks,
        "encoding_version": ENCODING_VERSION,
    }


def prompt_embeddings(config: dict, pipe, caption: str):
    """Prepare positive and unconditional embeddings with matching shapes."""
    import torch

    device, dtype, _ = runtime_device_dtype(config, pipe._execution_device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    negative_text = config["generation"].get("negative_prompt", "")
    # The ordinary empty negative prompt follows SDXL's zero-embedding path.
    minimum_chunks = 1
    if negative_text:
        minimum_chunks = max(
            math.ceil(len(tokenizer(negative_text, add_special_tokens=False)["input_ids"]) / (tokenizer.model_max_length - 2))
            for tokenizer in tokenizers
        )
    positive, pooled, details = encode_caption(config, tokenizers, encoders, caption, device, dtype, minimum_chunks)
    if negative_text:
        negative, negative_pooled, _ = encode_caption(
            config, tokenizers, encoders, negative_text, device, dtype, details["chunks"]
        )
    else:
        negative, negative_pooled = torch.zeros_like(positive), torch.zeros_like(pooled)
    return {
        "prompt_embeds": positive, "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": negative, "negative_pooled_prompt_embeds": negative_pooled,
    }, details


def image_seed(sample_id: str, global_seed: int) -> int:
    """Use SHA-256 rather than Python's process-randomized built-in hash."""
    import hashlib

    return (int(global_seed) + int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)) % (2**32)


def result_directory(config: dict, group: str, split: str, seed: int) -> Path:
    if group not in {"sd", "sd_lora"} or split not in {"train", "val", "test"}:
        raise ValueError("Unknown experiment group or split")
    return run_root(config) / group / split / f"seed_{seed}"


def package_versions(names) -> dict:
    """Use installed distribution versions in experiment identity, not guesses."""
    return {name: importlib.metadata.version(name) for name in names}


def generation_protocol(config: dict, records: list[dict], split: str) -> dict:
    """Describe all conditions that must be identical across the two groups."""
    import torch

    device, _, precision = runtime_device_dtype(config)
    return {
        "version": GENERATION_VERSION, "model": model_lock(config),
        "dataset_revision": config["data"]["revision"], "split": split,
        "manifest_sha256": manifest_sha(config, split),
        "sample_ids_sha256": json_sha256([record["sample_id"] for record in records]),
        "captions_sha256": json_sha256([record["caption"] for record in records]),
        "generation": {key: value for key, value in config["generation"].items() if key != "seeds"},
        "text": config["text"],
        "encoding_version": ENCODING_VERSION, "runtime": config["runtime"],
        # 'auto' alone is insufficient: an A100 may select bf16 while a T4
        # selects fp16. Refuse mixing those populations or resume segments.
        "resolved_runtime": {
            "precision": precision, "device_type": device.type,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_cuda": torch.version.cuda,
        },
        "implementation_versions": package_versions(("torch", "diffusers", "transformers", "peft", "Pillow", "numpy")),
        "source_sha256": {name: file_sha256(Path(__file__).resolve().parent / name) for name in ("data.py", "sd.py")},
    }


def save_png(image, target: Path) -> None:
    """Publish an image only after Pillow has finished writing the PNG."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".png.part")
    image.convert("RGB").save(temporary, format="PNG")
    temporary.replace(target)


def read_generation_resume(path: Path) -> list[dict]:
    """Recover only an unfinished final JSONL append after job interruption.

    Earlier malformed rows, or malformed rows terminated by a newline, are
    treated as corruption and remain fatal. An unterminated, invalid final
    byte sequence can be a torn append: quarantine it and atomically republish
    the valid prefix. The associated orphan PNG will be regenerated with its
    original seed. Call this only while holding the generation-directory lock.
    """
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for index, raw_line in enumerate(lines):
        try:
            record = json.loads(raw_line.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("A completed generation record must be a JSON object.")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index != len(lines) - 1 or raw_line.endswith(b"\n"):
                raise ValueError(f"Corrupted generation manifest at line {index + 1}: {path}") from error
            quarantine = path.with_name(f"{path.name}.torn-{time.time_ns()}.bin")
            quarantine.write_bytes(raw_line)
            write_jsonl(path, records)
            print(f"Recovered an unfinished final manifest append; saved bytes to {quarantine}", flush=True)
            return records
        records.append(record)
    if lines and not lines[-1].endswith(b"\n"):
        # A complete object whose final newline was interrupted is valid, but
        # append the separator before any later completion record is written.
        with path.open("ab") as output:
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
    return records


def generate_images(config: dict, group: str, split: str = "test", adapter=None,
                    records: list[dict] | None = None, seed: int | None = None,
                    output_dir: Path | None = None, pipeline=None) -> list[dict]:
    """Generate one PNG per caption, with hash-checked resumability.

    Metadata is written before expensive work and each completed image gets a
    durable JSONL record. A crash between image publication and its JSONL append
    leaves an orphan, which is regenerated with the SAME seed on resume. A
    recorded but corrupted image raises an error rather than silently changing
    the evaluation population. Generation is sequential by design.
    """
    import torch

    if group == "sd" and adapter is not None:
        raise ValueError("The original SD baseline must not load adapter weights.")
    if group == "sd_lora" and adapter is None:
        raise ValueError("The LoRA group requires an explicit adapter checkpoint.")
    records = load_split(config, split) if records is None else records
    if not records or len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Generation requires a non-empty set of unique sample IDs.")
    seed = int(config["generation"]["seeds"][0] if seed is None else seed)
    folder = Path(output_dir) if output_dir is not None else result_directory(config, group, split, seed)
    identity = adapter_identity(config, adapter)
    protocol = generation_protocol(config, records, split)
    signature = {"group": group, "seed": seed, "protocol": protocol, "adapter": identity}
    signature_hash = json_sha256(signature)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "generation_manifest.jsonl"
    with exclusive_file(folder / ".generation.lock", timeout=5.0):
        info_path = folder / "generation_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info["signature_sha256"] != signature_hash:
                raise ValueError(f"Existing images use a different generation protocol: {folder}")
        else:
            if manifest_path.exists():
                raise ValueError("A generation manifest without protocol metadata cannot be trusted.")
            write_json(info_path, {"signature_sha256": signature_hash, **signature})
        completed = {}
        if manifest_path.exists():
            for item in read_generation_resume(manifest_path):
                if item["sample_id"] in completed:
                    raise ValueError("Duplicate completed sample ID in generation manifest.")
                completed[item["sample_id"]] = item
        expected = {record["sample_id"]: record for record in records}
        if not set(completed).issubset(expected):
            raise ValueError("Generation directory contains samples outside the selected manifest.")
        for sample_id, item in completed.items():
            image_path = folder / item["image"]
            if item["caption"] != expected[sample_id]["caption"] or item["signature_sha256"] != signature_hash:
                raise ValueError(f"Completed sample metadata changed: {sample_id}")
            if not image_path.is_file() or file_sha256(image_path) != item["image_sha256"]:
                raise ValueError(f"Missing/corrupted completed PNG: {image_path}")
        missing = [record for record in records if record["sample_id"] not in completed]
        own_pipeline = pipeline is None
        if missing and pipeline is None:
            pipeline = build_pipeline(config, adapter)
        started = time.monotonic()
        if missing:
            if not (folder / "environment.json").exists():
                save_run_metadata(config, folder / "environment.json", {"signature_sha256": signature_hash})
            device, _, _ = runtime_device_dtype(config, pipeline._execution_device)
            with manifest_path.open("a", encoding="utf-8", newline="\n") as output:
                for number, record in enumerate(missing, 1):
                    actual_seed = image_seed(record["sample_id"], seed)
                    # A CPU generator yields the same starting noise construction
                    # regardless of CUDA device numbering. It does not promise
                    # identical final pixels across GPU types or library versions.
                    generator = torch.Generator(device="cpu").manual_seed(actual_seed)
                    with torch.inference_mode():
                        embeddings, text_details = prompt_embeddings(config, pipeline, record["caption"])
                        image = pipeline(
                            **embeddings, height=int(config["generation"]["height"]),
                            width=int(config["generation"]["width"]),
                            num_inference_steps=int(config["generation"]["steps"]),
                            guidance_scale=float(config["generation"]["guidance_scale"]),
                            generator=generator, num_images_per_prompt=1,
                        ).images[0]
                    target = folder / "images" / f"{record['sample_id']}.png"
                    save_png(image, target)
                    item = {
                        "sample_id": record["sample_id"], "caption": record["caption"],
                        "group": group, "global_seed": seed, "seed": actual_seed,
                        "image": f"images/{target.name}", "image_sha256": file_sha256(target),
                        "signature_sha256": signature_hash, "text_encoding": text_details,
                        "all_black": all(maximum == 0 for minimum, maximum in image.convert("RGB").getextrema()),
                    }
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    completed[record["sample_id"]] = item
                    print(f"{group} {split}: {len(completed)}/{len(records)} images", flush=True)
                    del embeddings
        write_json(folder / "generation_summary.json", {
            "samples": len(records), "new_images": len(missing),
            "new_images_seconds": round(time.monotonic() - started, 3),
            "all_black_images": sum(bool(item["all_black"]) for item in completed.values()),
            "complete": len(completed) == len(records),
        })
        if own_pipeline and pipeline is not None:
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return [completed[record["sample_id"]] for record in records]


def validate_generation(config: dict, group: str, split: str, seed: int, adapter=None,
                        records: list[dict] | None = None, output_dir: Path | None = None):
    """Require an exact sample population before computing any headline score."""
    from PIL import Image

    records = load_split(config, split) if records is None else records
    folder = Path(output_dir) if output_dir is not None else result_directory(config, group, split, seed)
    info = json.loads((folder / "generation_info.json").read_text(encoding="utf-8"))
    expected_signature = {
        "group": group, "seed": seed, "protocol": generation_protocol(config, records, split),
        "adapter": adapter_identity(config, adapter),
    }
    if info["signature_sha256"] != json_sha256(expected_signature):
        raise ValueError("Generation metadata does not match the selected group/config/adapter.")
    manifest = read_jsonl(folder / "generation_manifest.jsonl")
    by_id = {item["sample_id"]: item for item in manifest}
    expected_ids = {record["sample_id"] for record in records}
    if len(by_id) != len(manifest) or set(by_id) != expected_ids:
        raise ValueError("Missing/duplicate/unexpected generated sample IDs; incomplete results cannot be scored.")
    expected_files = {f"{sample_id}.png" for sample_id in expected_ids}
    actual_files = {path.name for path in (folder / "images").iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("The image folder contains missing or extra files; FID would use a different population.")
    paths = []
    for record in records:
        item = by_id[record["sample_id"]]
        if item["caption"] != record["caption"] or item["signature_sha256"] != info["signature_sha256"]:
            raise ValueError(f"Caption/protocol mismatch for {record['sample_id']}")
        if item["group"] != group or item["seed"] != image_seed(record["sample_id"], seed):
            raise ValueError(f"Group/seed mismatch for {record['sample_id']}")
        path = (folder / item["image"]).resolve()
        if path.parent != (folder / "images").resolve() or path.name != f"{record['sample_id']}.png":
            raise ValueError("Generated image path does not match its sample ID.")
        if file_sha256(path) != item["image_sha256"]:
            raise ValueError(f"Generated image checksum mismatch: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (int(config["generation"]["width"]), int(config["generation"]["height"])):
                raise ValueError(f"Unexpected generated image dimensions: {path}")
        paths.append(path)
    return records, paths, info


def compute_clipscore(config: dict, image_paths: list[Path], records: list[dict],
                      output_csv: Path | None = None) -> dict:
    """Compute 100 * max(0, cosine(CLIP image, CLIP text)) per sample.

    CLIPScore evaluates an image against its input caption, not against the
    reference photograph. Image and text features are L2-normalized before the
    dot product. The evaluator is deliberately independent of SDXL's encoders.
    Its own fixed context window is retained for the standard metric, with all
    truncated evaluator captions explicitly recorded. Long-prompt training
    chunking does not secretly redefine the reported evaluation metric.
    """
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if len(image_paths) != len(records) or not records:
        raise ValueError("CLIPScore requires one generated image for each non-empty caption record.")
    device, _, _ = runtime_device_dtype(config)
    lock = model_lock(config, "clip")
    kwargs = {
        "revision": lock["revision"],
        "cache_dir": str(resolve_path(config, config["paths"]["cache"]) / "huggingface"),
        "local_files_only": config["runtime"].get("local_files_only", False),
    }
    # OpenAI's official CLIP repository publishes a PyTorch .bin state dict.
    # Forcing safetensors at a locked commit would fail on that repository.
    # Keep the immutable SHA, allow the published format, and use the weights-
    # only loader provided by the reference PyTorch 2.8 environment.
    model = CLIPModel.from_pretrained(
        lock["repo_id"], use_safetensors=None, weights_only=True, **kwargs
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained(lock["repo_id"], **kwargs)
    context_limit = int(model.config.text_config.max_position_embeddings)
    batch_size = int(config["evaluation"]["clip_batch_size"])
    if batch_size < 1:
        raise ValueError("evaluation.clip_batch_size must be positive")
    policy = config["evaluation"].get("clip_long_caption_policy", "truncate_and_report")
    if policy not in {"error", "truncate_and_report"}:
        raise ValueError("Unknown evaluator long-caption policy.")
    rows, truncated = [], []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            selected = records[start:start + batch_size]
            texts = [record["caption"] for record in selected]
            lengths = [len(ids) for ids in processor.tokenizer(texts, truncation=False)["input_ids"]]
            long_ids = [record["sample_id"] for record, length in zip(selected, lengths) if length > context_limit]
            if long_ids and policy == "error":
                raise ValueError(f"Evaluation captions exceed the CLIP context window: {long_ids[:5]}")
            truncated.extend(long_ids)
            images = []
            for path in image_paths[start:start + batch_size]:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(
                text=texts, images=images, return_tensors="pt", padding=True,
                truncation=True, max_length=context_limit,
            ).to(device)
            image_features = functional.normalize(model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
            text_features = functional.normalize(model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ), dim=-1)
            scores = (100.0 * (image_features * text_features).sum(dim=-1).clamp_min(0)).cpu().tolist()
            for record, score, length in zip(selected, scores, lengths):
                rows.append({
                    "sample_id": record["sample_id"], "clipscore": float(score),
                    "evaluator_token_count": length, "evaluator_truncated": length > context_limit,
                })
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "clipscore": statistics.fmean(row["clipscore"] for row in rows),
        "n": len(rows), "clip_model": lock, "context_limit": context_limit,
        "evaluator_truncated_count": len(truncated), "evaluator_truncated_sample_ids": truncated,
        "formula": "mean(100 * max(0, cosine(normalized_image, normalized_text)))",
    }
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def compute_fid(config: dict, image_paths: list[Path], records: list[dict], split: str) -> dict:
    """Compare full image populations using clean-fid's Inception features.

    Real photographs are exported as oriented RGB PNGs at their native sizes.
    The clean-fid package owns resizing/quantization for both populations. Real
    feature statistics are cached under a key containing the exact data IDs,
    image hashes, preprocessing convention, and metric implementation version.
    The cache stores float64 moments, not a score against a particular model.
    """
    import numpy as np
    import torch
    from cleanfid import fid
    from cleanfid.features import build_feature_extractor

    if len(records) < 2 or len(image_paths) != len(records):
        raise ValueError("FID needs at least two real and two generated images with matching populations.")
    settings = config["evaluation"]
    mode = settings["fid_mode"]
    feature_name = settings["fid_feature_model"]
    if mode != "clean" or feature_name != "inception_v3":
        raise ValueError("This project's FID protocol fixes clean mode and Inception v3 features.")
    device, _, _ = runtime_device_dtype(config)
    signature = {
        "dataset_revision": config["data"]["revision"], "split": split,
        "manifest_sha256": manifest_sha(config, split),
        "sample_images": [(record["sample_id"], record["image_sha256"]) for record in records],
        "preprocessing": "exif-transpose-rgb-native-size-png-v1",
        "clean_fid_version": importlib.metadata.version("clean-fid"),
        "pillow_version": importlib.metadata.version("Pillow"),
        "mode": mode, "feature_model": feature_name,
        "implementation_versions": package_versions(("torch", "torchvision", "numpy", "scipy", "clean-fid", "Pillow")),
        "runtime": config["runtime"],
        "resolved_device": {
            "type": device.type, "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_cuda": torch.version.cuda,
        },
        "source_sha256": {name: file_sha256(Path(__file__).resolve().parent / name) for name in ("data.py", "sd.py")},
    }
    cache_key = json_sha256(signature)
    cache = resolve_path(config, config["paths"]["cache"]) / "fid" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    stats_path = cache / "real_statistics.npz"
    stats_info = cache / "real_statistics.json"
    extractor = build_feature_extractor(mode, device, use_dataparallel=False, model_name=feature_name)
    feature_options = {
        "model": extractor, "device": device, "mode": mode,
        "batch_size": int(settings["fid_batch_size"]), "num_workers": int(settings["fid_num_workers"]),
    }
    with exclusive_file(cache / ".cache.lock", timeout=3600.0):
        if stats_path.exists() and stats_info.exists():
            info = json.loads(stats_info.read_text(encoding="utf-8"))
            if info["key"] != cache_key or info["statistics_sha256"] != file_sha256(stats_path):
                raise ValueError("Cached real FID moments failed identity/integrity validation.")
            with np.load(stats_path, allow_pickle=False) as saved:
                real_mean, real_covariance = saved["mean"].copy(), saved["covariance"].copy()
        else:
            image_root = cache / "real_images"
            image_root.mkdir(parents=True, exist_ok=True)
            store = ParquetImageStore(
                resolve_path(config, config["paths"]["raw"]),
                max_cached_row_groups=int(config["data"]["row_group_cache"]),
            )
            # Exporting reference images is resumable. Every image is rewritten
            # from its pinned source before a previously absent stats cache is
            # created, so partial or user-edited exports cannot poison moments.
            for index, record in enumerate(records, 1):
                save_png(store.get_image(record), image_root / f"{record['sample_id']}.png")
                if index % 500 == 0:
                    print(f"Exported {index}/{len(records)} real FID images", flush=True)
            expected = {f"{record['sample_id']}.png" for record in records}
            if {path.name for path in image_root.iterdir()} != expected:
                raise ValueError("Unexpected files in the FID reference image directory.")
            real_features = fid.get_folder_features(str(image_root), **feature_options, description="FID real")
            if real_features.shape != (len(records), 2048):
                raise ValueError("Unexpected real Inception feature population/dimension.")
            real_mean = np.mean(real_features, axis=0, dtype=np.float64)
            real_covariance = np.cov(real_features.astype(np.float64), rowvar=False)
            temporary = cache / "real_statistics.part.npz"
            np.savez_compressed(temporary, mean=real_mean, covariance=real_covariance)
            temporary.replace(stats_path)
            write_json(stats_info, {"key": cache_key, "signature": signature, "n": len(records), "statistics_sha256": file_sha256(stats_path)})
            del real_features, store
    # Generated paths have already been checked against the exact manifest.
    # Calling the file-list API avoids globbing any unrelated output images.
    fake_features = fid.get_files_features(
        [str(path) for path in image_paths], **feature_options, description="FID generated"
    )
    if fake_features.shape != (len(records), 2048):
        raise ValueError("Unexpected generated Inception feature population/dimension.")
    fake_mean = np.mean(fake_features, axis=0, dtype=np.float64)
    fake_covariance = np.cov(fake_features.astype(np.float64), rowvar=False)
    if real_mean.shape != (2048,) or real_covariance.shape != (2048, 2048):
        raise ValueError("Cached Inception moments have incompatible dimensions.")
    value = float(fid.frechet_distance(real_mean, real_covariance, fake_mean, fake_covariance))
    if not math.isfinite(value):
        raise ValueError("FID calculation returned a non-finite value.")
    del extractor, fake_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"fid": value, "n_real": len(records), "n_generated": len(records), "real_cache_key": cache_key, "mode": mode, "feature_model": feature_name}


def evaluate_images(config: dict, group: str, split: str = "test", adapter=None,
                     records: list[dict] | None = None, seed: int | None = None,
                     output_dir: Path | None = None, include_fid: bool = True) -> dict:
    """Validate outputs, score them, and preserve enough metadata for comparison."""
    seed = int(config["generation"]["seeds"][0] if seed is None else seed)
    records, images, info = validate_generation(config, group, split, seed, adapter, records, output_dir)
    folder = Path(output_dir) if output_dir is not None else result_directory(config, group, split, seed)
    score_path = folder / "clipscore_per_sample.csv"
    clip = compute_clipscore(config, images, records, score_path)
    score_sha256 = file_sha256(score_path)
    fid_result = compute_fid(config, images, records, split) if include_fid else None
    scoring_protocol = evaluation_protocol(config, split, include_fid)
    result = {
        "group": group, "split": split, "seed": seed, "n": len(records),
        "clipscore": clip["clipscore"], "fid": None if fid_result is None else fid_result["fid"],
        "clip_details": clip, "fid_details": fid_result,
        "generation_signature_sha256": info["signature_sha256"], "protocol": info["protocol"],
        "generation_manifest_sha256": file_sha256(folder / "generation_manifest.jsonl"),
        "clipscore_per_sample_sha256": score_sha256,
        "evaluation_protocol": scoring_protocol,
        "evaluation_protocol_sha256": json_sha256(scoring_protocol),
        "adapter": info["adapter"], "diagnostic_only": len(records) != len(load_split(config, split)),
    }
    write_json(folder / "metrics.json", result)
    print(json.dumps({key: result[key] for key in ("group", "split", "seed", "n", "clipscore", "fid", "diagnostic_only")}, indent=2), flush=True)
    return result


def evaluation_protocol(config: dict, split: str, include_fid: bool = True) -> dict:
    """Bind scores to the evaluator identity as well as the generated images."""
    return {
        "version": "food500cap-scoring-v1", "clip_model": model_lock(config, "clip"),
        "settings": config["evaluation"], "include_fid": include_fid,
        "manifest_sha256": manifest_sha(config, split),
        "clip_formula": "mean(100 * max(0, cosine(normalized_image, normalized_text)))",
        "fid_preprocessing": "exif-transpose-rgb-native-size-png-v1",
        "implementation_versions": package_versions(("torch", "torchvision", "transformers", "clean-fid", "Pillow", "numpy", "scipy")),
        "source_sha256": {name: file_sha256(Path(__file__).resolve().parent / name) for name in ("data.py", "sd.py")},
    }


def _read_comparison_scores(folder: Path, records: list[dict], expected_hash: str | None) -> dict[str, float]:
    """Join the scored images to the validated population by stable sample ID."""
    path = folder / "clipscore_per_sample.csv"
    # The aggregate results and ranked examples must come from the same
    # scoring pass. A copied or edited CSV must not silently change the ranks.
    if expected_hash is None or not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"Per-image CLIPScore CSV is missing or stale; re-evaluate {folder}")
    scores = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id, score = row["sample_id"], float(row["clipscore"])
            if sample_id in scores or not math.isfinite(score):
                raise ValueError(f"Duplicate sample ID or non-finite CLIPScore in {path}: {sample_id}")
            scores[sample_id] = score
    if scores.keys() != {record["sample_id"] for record in records}:
        raise ValueError(f"CLIPScore sample IDs do not match the validated images: {path}")
    return scores


def _save_comparison_grid(store: ParquetImageStore, image_dirs: dict[str, Path], records: list[dict],
                          scores: dict[str, dict[str, float]], title: str, target: Path) -> None:
    """Show the real image and both generated images for each selected caption."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    import textwrap

    cell_width, image_height, label_height, header_height = 320, 280, 100, 60
    canvas = Image.new("RGB", (3 * cell_width, header_height + len(records) * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default(size=14)
    draw.text((8, 8), title, fill="black", font=font)
    for column, heading in enumerate(("Real reference", "SDXL", "SDXL + LoRA")):
        draw.text((column * cell_width + 8, 34), heading, fill="black", font=font)
    for row_index, record in enumerate(records):
        sample_id = record["sample_id"]
        images = [store.get_image(record)]
        for group in ("sd", "sd_lora"):
            with Image.open(image_dirs[group] / f"{sample_id}.png") as source:
                images.append(source.convert("RGB"))
        top = header_height + row_index * (image_height + label_height)
        for column, image in enumerate(images):
            thumb = ImageOps.contain(image, (cell_width - 12, image_height - 12))
            canvas.paste(thumb, (column * cell_width + (cell_width - thumb.width) // 2, top + (image_height - thumb.height) // 2))
        label = f"{sample_id}   |   CLIPScore: SDXL {scores['sd'][sample_id]:.2f}   |   SDXL + LoRA {scores['sd_lora'][sample_id]:.2f}"
        draw.text((8, top + image_height), label, fill="black", font=font)
        # Shorten only the display text. Model input and the split manifest
        # still use the complete original caption.
        caption = textwrap.fill(record["caption"], width=115, max_lines=4, placeholder=" ...")
        draw.multiline_text((8, top + image_height + 20), caption, fill="black", font=font, spacing=2)
    save_png(canvas, target)


def save_comparison(config: dict, split: str = "test") -> None:
    """Save aggregate metrics, a fixed sample, and examples at each score extreme."""
    seeds = [int(seed) for seed in config["generation"]["seeds"]]
    table, paired_protocols = [], {}
    example_scores: dict[str, dict[str, float]] = {}
    for seed in seeds:
        pair, scoring_pair, fid_reference_pair = [], [], []
        for group in ("sd", "sd_lora"):
            folder = result_directory(config, group, split, seed)
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            records, _, info = validate_generation(
                config, group, split, seed,
                None if group == "sd" else metrics["adapter"]["path"],
            )
            if metrics["generation_signature_sha256"] != info["signature_sha256"] or metrics["diagnostic_only"]:
                raise ValueError("Stale or diagnostic metrics cannot enter the final comparison.")
            if metrics["generation_manifest_sha256"] != file_sha256(folder / "generation_manifest.jsonl"):
                raise ValueError("Generated images changed after scoring; recompute metrics before reporting.")
            current_scoring_hash = json_sha256(evaluation_protocol(config, split, True))
            if metrics.get("evaluation_protocol_sha256") != current_scoring_hash:
                raise ValueError("Metrics use a different evaluator/configuration/version; rescore both groups.")
            if metrics["n"] != len(records) or metrics["protocol"] != info["protocol"]:
                raise ValueError("Stored metrics do not describe the validated generated population.")
            if seed == seeds[0]:
                example_scores[group] = _read_comparison_scores(folder, records, metrics.get("clipscore_per_sample_sha256"))
            pair.append(json_sha256(metrics["protocol"]))
            scoring_pair.append(metrics["evaluation_protocol_sha256"])
            fid_reference_pair.append(metrics["fid_details"]["real_cache_key"])
            table.append({key: metrics[key] for key in ("group", "seed", "n", "clipscore", "fid")})
        if pair[0] != pair[1] or scoring_pair[0] != scoring_pair[1] or fid_reference_pair[0] != fid_reference_pair[1]:
            raise ValueError("The two groups used different prompts, preprocessing, model, or generation parameters.")
        paired_protocols[str(seed)] = pair[0]
    if not table:
        raise ValueError("No generation seeds were configured.")
    destination = run_root(config) / "comparison" / split
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary = {}
    for group in ("sd", "sd_lora"):
        group_rows = [row for row in table if row["group"] == group]
        summary[group] = {}
        for metric in ("clipscore", "fid"):
            values = [row[metric] for row in group_rows]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("Final comparison requires finite CLIPScore and FID for every seed.")
            summary[group][metric] = {"mean": statistics.fmean(values), "std_across_generation_seeds": statistics.stdev(values) if len(values) > 1 else None}
    write_json(destination / "summary.json", {"groups": summary, "paired_protocols": paired_protocols, "training_seed": config["experiment"]["train_seed"], "note": "Generation-seed variation is not repeated-training uncertainty."})
    # Keep the fixed random grid as a score-independent view of both groups.
    # The four ranked grids serve a different purpose: inspecting the extremes
    # of each group's image/text alignment on the first configured seed.
    candidates = load_split(config, split)
    count = min(int(config["evaluation"]["comparison_samples"]), len(candidates))
    chosen = sorted(random.Random(0).sample(candidates, count), key=lambda record: record["sample_id"])
    seed = seeds[0]
    image_dirs = {group: result_directory(config, group, split, seed) / "images" for group in ("sd", "sd_lora")}
    ranked_examples = {}
    store = ParquetImageStore(resolve_path(config, config["paths"]["raw"]), max_cached_row_groups=int(config["data"]["row_group_cache"]))
    try:
        _save_comparison_grid(store, image_dirs, chosen, example_scores,
                              f"Fixed random examples | seed {seed}", destination / "comparison.png")
        # Take opposite ends of one ordering, with sample ID as a stable
        # tiebreaker. On small splits, cap both ends so they cannot overlap.
        ranked_count = min(count, len(candidates) // 2)
        if ranked_count:
            for group, label in (("sd", "SDXL"), ("sd_lora", "SDXL + LoRA")):
                group_scores = example_scores[group]
                ranked = sorted(candidates, key=lambda record: (group_scores[record["sample_id"]], record["sample_id"]))
                for level, selected in (("high", ranked[-ranked_count:][::-1]), ("low", ranked[:ranked_count])):
                    filename = f"{group}_{level}_clipscore.png"
                    _save_comparison_grid(store, image_dirs, selected, example_scores,
                                          f"{label} | {level.title()} CLIPScore examples | seed {seed}", destination / filename)
                    ranked_examples[filename] = [record["sample_id"] for record in selected]
    finally:
        store.close()
    write_json(destination / "comparison_samples.json", {
        "seed": seed, "sample_ids": [record["sample_id"] for record in chosen], "ranked_by_clipscore": ranked_examples,
    })
    print(f"Comparison written to {destination}", flush=True)


def inference_commands(config: dict, group: str, command: str, split: str, adapter=None, limit: int | None = None):
    """Shared CLI dispatcher; limits create separate, explicitly diagnostic runs."""
    records = load_split(config, split)
    if limit is not None:
        if limit < 2:
            raise ValueError("--limit must be at least 2 for diagnostic FID.")
        records = records[:limit]
    for seed in config["generation"]["seeds"]:
        output_dir = None
        if limit is not None:
            output_dir = run_root(config) / "diagnostics" / f"n_{len(records)}" / group / split / f"seed_{seed}"
        if command in {"generate", "all"}:
            generate_images(config, group, split, adapter, records, int(seed), output_dir)
        if command in {"evaluate", "all"}:
            evaluate_images(config, group, split, adapter, records, int(seed), output_dir)


def report_when_ready(config: dict) -> None:
    """Let whichever group finishes last write the formal test comparison."""
    # A metrics file marks a completed scoring pass. Its existence is only a
    # readiness check; save_comparison still validates every file and protocol.
    for group, script in (("sd", "sd.py"), ("sd_lora", "sd_lora.py")):
        if any(not (result_directory(config, group, "test", int(seed)) / "metrics.json").is_file()
               for seed in config["generation"]["seeds"]):
            print(f"Run {script} to finish the two-group comparison.", flush=True)
            return
    # If the groups finish at nearly the same time, serialize report writes so
    # their CSV and image files cannot be written by two processes at once.
    with exclusive_file(run_root(config) / "comparison" / "test" / ".report.lock"):
        save_comparison(config, "test")


def main() -> None:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    if len(sys.argv) == 1:
        # A plain IDE Run launches the complete baseline with the same paths,
        # settings, and validation as the explicit CLI's `all` command.
        config = load_config(default_config)
        print("Running the original SDXL baseline on the full test split...", flush=True)
        inference_commands(config, "sd", "all", "test")
        report_when_ready(config)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "evaluate", "report", "all"))
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--limit", type=int, help="Use a separate diagnostic output population; omit for final scores.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "report":
        if args.limit is not None:
            parser.error("report requires full-population metrics, so --limit is not supported")
        save_comparison(config, args.split)
    else:
        inference_commands(config, "sd", args.command, args.split, limit=args.limit)


if __name__ == "__main__":
    main()
