"""Food500Cap acquisition, provenance, leakage-safe splits, and lazy image loading.

Running this file with no arguments audits the local dataset and prepares the
three splits. Importing this module never downloads data, decodes images,
imports a training framework, or changes an artifact. The raw Parquet rows and
their complete captions remain immutable. Processed JSONL files only point back
to those rows; they do not contain recompressed images.

Examples, run deliberately by the user from the directory containing config.yaml:
    python data.py download --config config.yaml
    python data.py audit --config config.yaml
    python data.py verify --config config.yaml
    python data.py prepare --config config.yaml
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Iterator
from urllib.parse import quote
from urllib.request import Request, urlopen
import warnings


REPO_ID = "advancedcv/Food500Cap"
REVISION = "1b2e0cb098ef3d58584f80bde015207c5a3d00c7"
SPLIT_NAMES = ("train", "val", "test")
SOURCE_FIELDS = (
    "sample_id", "source_split", "shard", "row_in_shard", "category_raw",
    "caption", "image_sha256", "pixel_sha256", "width", "height",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read YAML without loading any model or image library.

    Relative artifact paths are always relative to the configuration file,
    which makes the same command usable from a terminal or a cluster job.
    Internal location keys are excluded by callers when archiving settings.
    """
    import yaml

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("The configuration must be a YAML mapping")
    for section in ("paths", "data", "model", "experiment"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping")
    for key in ("raw", "processed", "reports", "cache"):
        if not isinstance(cfg["paths"].get(key), str) or not cfg["paths"][key].strip():
            raise ValueError(f"paths.{key} must be a nonempty path string")
    if cfg["data"].get("repo_id") != REPO_ID or cfg["data"].get("revision") != REVISION:
        raise ValueError(f"This experiment requires {REPO_ID} at pinned revision {REVISION}")
    fraction = cfg["data"].get("validation_fraction", 0.1)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("data.validation_fraction must be between zero and one")
    cfg["data"]["validation_fraction"] = float(fraction)
    for key, default in (("split_seed", 2026), ("row_group_cache", 2)):
        value = cfg["data"].get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"data.{key} must be an integer")
        if key == "row_group_cache" and value < 1:
            raise ValueError("data.row_group_cache must be positive")
        cfg["data"][key] = value
    if cfg["model"].get("family") != "sdxl":
        raise ValueError("model.family must be 'sdxl'; SDXL conditioning is used throughout")
    if not isinstance(cfg["experiment"].get("output_dir"), str) or not cfg["experiment"]["output_dir"].strip():
        raise ValueError("experiment.output_dir must be a nonempty path string")
    cfg["_root"] = str(config_path.parent)
    cfg["_config_path"] = str(config_path)
    # Data, reports, and processed manifests must occupy distinct directories.
    roots = {key: resolve_path(cfg, cfg["paths"][key]) for key in ("raw", "processed", "reports")}
    for first, first_path in roots.items():
        for second, second_path in roots.items():
            if first != second and (first_path == second_path or first_path.is_relative_to(second_path)):
                raise ValueError(f"paths.{first} and paths.{second} must not be nested or equal")
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    """Resolve a configured filesystem path without creating it."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(cfg["_root"]) / path).resolve()


def _path(cfg: dict[str, Any], name: str) -> Path:
    return resolve_path(cfg, cfg["paths"][name])


def _safe_child(root: Path, relative: str) -> Path:
    """Reject upstream or manifest paths that escape their artifact directory."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a relative artifact path, received {relative!r}")
    target = (root / relative).resolve()
    if target == root.resolve() or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its root: {relative!r}")
    return target


def _atomic_text(path: Path, lines: Iterable[str]) -> None:
    """Publish a complete file with a same-directory atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: str | Path, value: Any) -> None:
    _atomic_text(Path(path), [json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"])


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    _atomic_text(Path(path), (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            records.append(row)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    """Hash canonical JSON, independent of whitespace and mapping insertion order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if metadata.get("sha") != REVISION or metadata.get("id") != REPO_ID:
        raise ValueError("Source metadata does not describe the pinned Food500Cap snapshot")
    files = [item for item in metadata["siblings"]
             if item["rfilename"] in {"README.md", ".gitattributes"}
             or item["rfilename"].endswith(".parquet")]
    names = [item["rfilename"] for item in files]
    if len(names) != len(set(names)) or not any(name.endswith(".parquet") for name in names):
        raise ValueError("Invalid or empty upstream file inventory")
    for item in files:
        if not isinstance(item.get("size"), int) or item["size"] < 0:
            raise ValueError(f"Missing upstream size for {item['rfilename']}")
        expected = item.get("lfs", {}).get("sha256") or item.get("blobId")
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", expected):
            raise ValueError(f"Missing upstream object hash for {item['rfilename']}")
    return sorted(files, key=lambda item: item["rfilename"])


def _verify_upstream_file(path: Path, item: dict[str, Any]) -> str:
    """Verify LFS SHA-256, or the Git blob SHA-1 for ordinary repository files."""
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Size mismatch: {item['rfilename']}")
    sha256 = hashlib.sha256()
    git_digest = hashlib.sha1(f"blob {item['size']}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_digest.update(chunk)
    lfs_hash = item.get("lfs", {}).get("sha256")
    actual, expected = (sha256.hexdigest(), lfs_hash) if lfs_hash else (git_digest.hexdigest(), item["blobId"])
    if actual != expected:
        raise ValueError(f"Upstream object hash mismatch: {item['rfilename']}")
    return sha256.hexdigest()


def download_dataset(cfg: dict[str, Any], workers: int | None = None) -> dict[str, Any]:
    """Resume downloads with the standard library and verify every complete file.

    A server that ignores Range starts a fresh partial file.  A completed but
    corrupt partial is discarded before retrying; existing final raw files are
    checked and never silently overwritten when their hash is wrong.
    """
    raw = _path(cfg, "raw")
    count = workers if workers is not None else cfg["data"].get("download_workers", 4)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Download worker count must be positive")
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/revision/{REVISION}?blobs=true"
    with urlopen(Request(url, headers={"User-Agent": "Food500Cap-final-project/2.0"}), timeout=60) as response:
        metadata = json.load(response)
    files = _selected_files(metadata)
    raw.mkdir(parents=True, exist_ok=True)
    write_json(raw / "source_metadata.json", metadata)
    progress: dict[str, int] = {}
    lock = threading.Lock()

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        name = item["rfilename"]
        target = _safe_child(raw, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists():
            checksum = _verify_upstream_file(target, item)
            with lock:
                progress[name] = item["size"]
            return {"path": name, "size": item["size"], "sha256": checksum, "verified": True}
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item["size"]:
                raise ValueError(f"Oversized partial download: {name}")
            try:
                if offset != item["size"]:
                    address = (f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/"
                               f"{quote(name, safe='/')}?download=true&attempt={attempt}")
                    headers = {"User-Agent": "Food500Cap-final-project/2.0"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    with urlopen(Request(address, headers=headers), timeout=60) as response:
                        if response.status not in (200, 206):
                            raise ValueError(f"Unexpected HTTP status {response.status}")
                        append = bool(offset and response.status == 206)
                        if response.status == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            expected_start = offset if append else 0
                            if not match or int(match[1]) != expected_start or int(match[3]) != item["size"]:
                                raise ValueError("Unexpected resumed Content-Range")
                        if not append:
                            offset = 0
                        with partial.open("ab" if append else "wb") as output:
                            while chunk := response.read(4 * 1024 * 1024):
                                output.write(chunk)
                                offset += len(chunk)
                                if offset > item["size"]:
                                    raise ValueError("Response exceeds the upstream file size")
                                with lock:
                                    progress[name] = offset
                checksum = _verify_upstream_file(partial, item)
                partial.replace(target)
                with lock:
                    progress[name] = item["size"]
                print(f"VERIFIED {name}", flush=True)
                return {"path": name, "size": item["size"], "sha256": checksum, "verified": True}
            except Exception as error:
                print(f"RETRY {name}: {type(error).__name__} ({attempt + 1}/6)", flush=True)
                if attempt == 5:
                    raise
                if partial.exists() and partial.stat().st_size >= item["size"]:
                    partial.unlink()
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError("Download attempts exhausted")

    total = sum(item["size"] for item in files)
    records = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=count) as executor:
        pending = {executor.submit(fetch, item) for item in files}
        while pending:
            complete, pending = wait(pending, timeout=10)
            for future in complete:
                records.append(future.result())
            with lock:
                completed = sum(progress.values())
            print(f"PROGRESS {completed / 1e9:.3f}/{total / 1e9:.3f} GB; "
                  f"{len(records)}/{len(files)} verified; {time.monotonic() - start:.0f}s", flush=True)
            write_json(raw / "download_progress.json", {"bytes": completed, "total_bytes": total,
                       "verified_files": len(records), "total_files": len(files)})
    manifest = {"repo_id": REPO_ID, "revision": REVISION,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_bytes": total, "files": sorted(records, key=lambda row: row["path"])}
    write_json(raw / "download_manifest.json", manifest)
    return manifest


def verify_raw(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read and hash all raw files against both upstream metadata and local receipt."""
    raw = _path(cfg, "raw")
    metadata = _read_json(raw / "source_metadata.json")
    files = _selected_files(metadata)
    receipt = _read_json(raw / "download_manifest.json")
    if receipt.get("repo_id") != REPO_ID or receipt.get("revision") != REVISION:
        raise ValueError("Download manifest has a different dataset identity or revision")
    received = {row["path"]: row for row in receipt["files"]}
    if len(received) != len(receipt["files"]) or set(received) != {item["rfilename"] for item in files}:
        raise ValueError("Download manifest and upstream file inventories differ")
    verified = []
    for item in files:
        name = item["rfilename"]
        checksum = _verify_upstream_file(_safe_child(raw, name), item)
        row = received[name]
        if row.get("sha256") != checksum or row.get("size") != item["size"] or row.get("verified") is not True:
            raise ValueError(f"Download manifest hash/size mismatch for {name}")
        verified.append({"path": name, "size": item["size"], "sha256": checksum})
    total = sum(row["size"] for row in verified)
    if receipt.get("total_bytes") != total:
        raise ValueError("Download manifest total_bytes is inconsistent")
    return {"repo_id": REPO_ID, "revision": REVISION, "files": verified,
            "source_metadata_sha256": file_sha256(raw / "source_metadata.json"),
            "download_manifest_sha256": file_sha256(raw / "download_manifest.json")}


def _summary(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {"min": ordered[0], "median": ordered[len(ordered) // 2],
            "p90": ordered[int((len(ordered) - 1) * 0.9)], "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 2)}


def _pixel_hash(image: Any) -> str:
    # The encoded-byte hash identifies a particular stored image object.  This
    # second hash identifies equal oriented RGB pixels, even if file metadata
    # or encoding differs.  Size is part of the hash to prevent shape ambiguity.
    width, height = image.size
    digest = hashlib.sha256(f"RGB:{width}x{height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def normalize_category(category: str) -> str:
    """Normalize only the stratification key; retain the raw category and caption."""
    return " ".join(category.split()).casefold()


def _exclusions(records: list[dict[str, Any]]) -> dict[str, Any]:
    test_by_hash: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if row["source_split"] == "test" and row.get("pixel_sha256"):
            test_by_hash[row["pixel_sha256"]].append(row["sample_id"])
    exclusions = []
    for row in records:
        digest = row.get("pixel_sha256")
        if row["source_split"] == "train" and digest in test_by_hash:
            exclusions.append({"sample_id": row["sample_id"],
                               "matching_test_sample_ids": sorted(test_by_hash[digest]),
                               "pixel_sha256": digest,
                               "reason": "exact_pixel_duplicate_of_test_image"})
    train_count = sum(row["source_split"] == "train" for row in records)
    return {"dataset_revision": REVISION, "source": "duplicate_groups.json",
            "policy": "Keep every upstream test row unchanged; exclude every matching train image before train/validation splitting.",
            "preserved_in_raw_and_source_index": True,
            "excluded_train_rows": len(exclusions), "remaining_train_candidates": train_count - len(exclusions),
            "exclusions": exclusions}


def audit_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Decode every image and validate every caption, while preserving all raw rows.

    Warnings such as malformed EXIF metadata are recorded separately from fatal
    decode errors.  EXIF orientation is applied before dimensions and RGB hashes
    are measured, matching both duplicate isolation and training image loading.
    """
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    provenance = verify_raw(cfg)
    raw, reports = _path(cfg, "raw"), _path(cfg, "reports")
    metadata = _read_json(raw / "source_metadata.json")
    expected = {row["name"]: row["num_examples"] for row in metadata["cardData"]["dataset_info"]["splits"]}
    records: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, Counter] = defaultdict(Counter)
    stats, failures, image_warnings = {}, [], []
    formats, modes = Counter(), Counter()
    widths, heights, characters, words = [], [], [], []
    start = time.monotonic()
    for split in ("train", "test"):
        split_count = 0
        shards = sorted((raw / "data").glob(f"{split}-*.parquet"))
        upstream_shards = {item["rfilename"] for item in metadata["siblings"]
                           if item["rfilename"].startswith(f"data/{split}-")
                           and item["rfilename"].endswith(".parquet")}
        if {f"data/{path.name}" for path in shards} != upstream_shards:
            raise ValueError(f"Missing or unexpected Parquet shards for {split}")
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            try:
                if set(parquet.schema_arrow.names) != {"image", "cat", "caption"}:
                    raise ValueError(f"Unexpected schema in {shard.name}")
                row_in_shard = 0
                for batch in parquet.iter_batches(batch_size=32):
                    for source in batch.to_pylist():
                        sample_id = f"{split}-{split_count:06d}"
                        category, caption = source["cat"], source["caption"]
                        row = {"sample_id": sample_id, "source_split": split,
                               "shard": f"data/{shard.name}", "row_in_shard": row_in_shard,
                               "category_raw": category, "caption": caption}
                        categories[split][str(category)] += 1
                        if not isinstance(caption, str) or not caption.strip():
                            failures.append({"sample_id": sample_id, "reason": "empty_or_invalid_caption"})
                        if not isinstance(category, str) or not category.strip():
                            failures.append({"sample_id": sample_id, "reason": "empty_or_invalid_category"})
                        if isinstance(caption, str):
                            characters.append(len(caption))
                            words.append(len(caption.split()))
                        captured = []
                        try:
                            with warnings.catch_warnings(record=True) as captured:
                                warnings.simplefilter("always")
                                encoded = source["image"]["bytes"]
                                if not encoded:
                                    raise ValueError("Embedded image bytes are missing")
                                row["image_sha256"] = hashlib.sha256(encoded).hexdigest()
                                with Image.open(BytesIO(encoded)) as original:
                                    formats[str(original.format)] += 1
                                    modes[original.mode] += 1
                                    original.load()
                                    image = ImageOps.exif_transpose(original).convert("RGB")
                                    width, height = image.size
                                    digest = _pixel_hash(image)
                                    row.update({"width": width, "height": height, "pixel_sha256": digest})
                                    widths.append(width)
                                    heights.append(height)
                                    groups[digest].append({"sample_id": sample_id, "split": split,
                                                           "category_raw": category})
                        except Exception as error:
                            failures.append({"sample_id": sample_id, "reason": "image_decode_error",
                                             "error_type": type(error).__name__, "message": str(error)})
                        for warning in captured:
                            image_warnings.append({"sample_id": sample_id,
                                                   "warning_type": warning.category.__name__,
                                                   "message": str(warning.message)})
                        records.append(row)
                        split_count += 1
                        row_in_shard += 1
                    if len(records) % 512 < 32:
                        print(f"AUDIT {len(records)} images; {time.monotonic() - start:.0f}s", flush=True)
                print(f"SHARD {shard.name}: {row_in_shard} rows", flush=True)
            finally:
                parquet.close()
        stats[split] = {"rows": split_count, "expected_rows": expected[split],
                        "categories_raw": len(categories[split]),
                        "category_counts": dict(sorted(categories[split].items()))}
        if split_count != expected[split]:
            failures.append({"split": split, "reason": "row_count_mismatch"})
    duplicates = [{"pixel_sha256": key, "members": members}
                  for key, members in sorted(groups.items()) if len(members) > 1]
    overlap = [group for group in duplicates
               if {member["split"] for member in group["members"]} == {"train", "test"}]
    all_categories = set(categories["train"]) | set(categories["test"])
    aliases: dict[str, list[str]] = defaultdict(list)
    for name in sorted(all_categories):
        aliases[normalize_category(name)].append(name)
    write_jsonl(reports / "source_index.jsonl", records)
    write_json(reports / "duplicate_groups.json", {"dataset_revision": REVISION, "groups": duplicates})
    write_json(reports / "train_exclusions.json", _exclusions(records))
    audit = {
        "dataset": REPO_ID, "revision": REVISION, "total_rows": len(records),
        "schema": ["image", "cat", "caption"], "splits": stats,
        "all_images_decoded": len(widths) == len(records), "failures": failures,
        "image_warning_count": len(image_warnings), "image_warnings": image_warnings,
        "unique_pixel_hashes": len(groups), "duplicate_pixel_groups": len(duplicates),
        "cross_split_duplicate_groups": len(overlap),
        "cross_split_duplicate_train_rows": sum(member["split"] == "train" for group in overlap for member in group["members"]),
        "cross_split_duplicate_test_rows": sum(member["split"] == "test" for group in overlap for member in group["members"]),
        "categories_raw_union": len(all_categories), "categories_normalized_union": len(aliases),
        "possible_case_whitespace_aliases": [names for names in aliases.values() if len(names) > 1],
        "image_width": _summary(widths), "image_height": _summary(heights),
        "image_formats": dict(formats), "image_modes": dict(modes),
        "images_with_short_side_below_512": sum(min(w, h) < 512 for w, h in zip(widths, heights)),
        "images_with_short_side_below_1024": sum(min(w, h) < 1024 for w, h in zip(widths, heights)),
        "caption_characters": _summary(characters), "caption_whitespace_words": _summary(words),
        "caption_token_lengths_checked": False,
        "duplicate_detection_scope": "SHA-256 of EXIF-oriented RGB pixels at original resolution; not perceptual near-duplicate detection",
        "source_index": "source_index.jsonl", "source_index_sha256": file_sha256(reports / "source_index.jsonl"),
        "duplicate_groups_sha256": file_sha256(reports / "duplicate_groups.json"),
        "train_exclusions_sha256": file_sha256(reports / "train_exclusions.json"),
        "raw_provenance": provenance, "duration_seconds": round(time.monotonic() - start, 1),
    }
    write_json(reports / "data_audit.json", audit)
    print(f"AUDIT rows={len(records)} failures={len(failures)} warnings={len(image_warnings)} "
          f"test-overlap train exclusions={audit['cross_split_duplicate_train_rows']}", flush=True)
    if failures:
        raise ValueError("Dataset audit found failures; inspect data_audit.json before preparing splits")
    return audit


def _validate_source_row(row: dict[str, Any]) -> None:
    missing = set(SOURCE_FIELDS) - set(row)
    if missing:
        raise ValueError(f"Incomplete source record {row.get('sample_id')}: {sorted(missing)}")
    if row["source_split"] not in ("train", "test"):
        raise ValueError("Unexpected upstream split")
    if not isinstance(row["sample_id"], str) or not re.fullmatch(r"(?:train|test)-\d{6}", row["sample_id"]):
        raise ValueError("Invalid source sample_id")
    if not row["sample_id"].startswith(row["source_split"] + "-"):
        raise ValueError("Sample identifier and upstream split disagree")
    for key in ("category_raw", "caption"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Invalid {key} in {row['sample_id']}")
    for key in ("image_sha256", "pixel_sha256"):
        if not isinstance(row[key], str) or not re.fullmatch(r"[a-f0-9]{64}", row[key]):
            raise ValueError(f"Invalid {key} in {row['sample_id']}")
    for key in ("row_in_shard", "width", "height"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < (0 if key == "row_in_shard" else 1):
            raise ValueError(f"Invalid {key} in {row['sample_id']}")


def _validated_source_records(cfg: dict[str, Any], provenance: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports = _path(cfg, "reports")
    audit = _read_json(reports / "data_audit.json")
    if audit.get("dataset") != REPO_ID or audit.get("revision") != REVISION:
        raise ValueError("The audit does not match the pinned dataset")
    if audit.get("failures") or audit.get("all_images_decoded") is not True:
        raise ValueError("A complete successful image/caption audit is required")
    # Older exploratory reports did not bind their derived index to the raw
    # snapshot. Row counts cannot prove that captions, row mappings, or pixel
    # groups were preserved. Require a fresh audit from this implementation.
    if audit.get("source_index_sha256") is None or audit.get("raw_provenance") is None:
        raise ValueError("Legacy audit lacks provenance hashes; run data.py audit before data.py prepare")
    index_hash = file_sha256(reports / "source_index.jsonl")
    if audit["source_index_sha256"] != index_hash:
        raise ValueError("The source index changed after its audit; rerun audit")
    if audit["raw_provenance"] != provenance:
        raise ValueError("Raw provenance changed after its audit; rerun audit")
    records = read_jsonl(reports / "source_index.jsonl")
    if len(records) != audit.get("total_rows"):
        raise ValueError("Source-index and audit row counts differ")
    counts, ids, locations = Counter(), set(), set()
    expected_files = {row["path"] for row in provenance["files"]}
    for row in records:
        _validate_source_row(row)
        _safe_child(_path(cfg, "raw"), row["shard"])
        if row["shard"] not in expected_files or not row["shard"].startswith(f"data/{row['source_split']}-"):
            raise ValueError("Source index references a different raw shard")
        location = (row["shard"], row["row_in_shard"])
        if row["sample_id"] in ids or location in locations:
            raise ValueError("Repeated source sample ID or Parquet row location")
        ids.add(row["sample_id"])
        locations.add(location)
        counts[row["source_split"]] += 1
    metadata = _read_json(_path(cfg, "raw") / "source_metadata.json")
    expected = {row["name"]: row["num_examples"] for row in metadata["cardData"]["dataset_info"]["splits"]}
    if dict(counts) != expected:
        raise ValueError("Source-index counts differ from pinned upstream metadata")
    for split, count in counts.items():
        if audit["splits"][split]["rows"] != count or audit["splits"][split]["expected_rows"] != count:
            raise ValueError("Audit split counts are inconsistent")
        if {row["sample_id"] for row in records if row["source_split"] == split} != {
            f"{split}-{number:06d}" for number in range(count)
        }:
            raise ValueError("Source sample IDs are not the complete upstream sequence")
    return records, audit


def _stratified_validation_groups(records: list[dict[str, Any]], fraction: float, seed: int) -> tuple[set[str], dict[str, Any]]:
    """Stratify whole pixel groups by their majority normalized category.

    A duplicate group may contain more than one category label.  Its majority
    normalized category defines one unambiguous stratum, with lexical tie-breaks.
    This decision affects allocation only: each row keeps its original category
    and complete caption.  Within each stratum, seeded shuffling selects roughly
    the requested fraction of groups, while retaining at least one training
    group.  A stratum with one eligible group therefore remains in training.
    """
    labels: dict[str, Counter] = defaultdict(Counter)
    class_groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        digest, category = row["pixel_sha256"], normalize_category(row["category_raw"])
        labels[digest][category] += 1
    for digest in sorted(labels):
        majority = min(labels[digest], key=lambda category: (-labels[digest][category], category))
        class_groups[majority].append(digest)
    rng = random.Random(seed)
    validation: set[str] = set()
    quotas = {}
    for category in sorted(class_groups):
        groups = list(class_groups[category])
        rng.shuffle(groups)
        count = min(len(groups) - 1, max(1, int(math.floor(len(groups) * fraction + 0.5)))) if len(groups) > 1 else 0
        validation.update(groups[:count])
        quotas[category] = {"eligible_groups": len(groups), "target_val_groups": count, "actual_val_groups": count}
    report = {"algorithm": "seeded per-class pixel-group shuffle with one training group retained per stratum",
              "group_category": "majority normalized category across group rows, with lexical tie-break",
              "quota_unit": "distinct oriented-RGB pixel groups per majority normalized category",
              "singleton_group_categories_kept_in_train": sorted(name for name, groups in class_groups.items() if len(groups) == 1),
              "multicategory_pixel_groups": sum(len(value) > 1 for value in labels.values()),
              "category_group_quotas": quotas}
    return validation, report


def _split_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": len(records), "pixel_groups": len({row["pixel_sha256"] for row in records}),
            "categories_normalized": len({row["category_normalized"] for row in records}),
            "category_counts": dict(sorted(Counter(row["category_normalized"] for row in records).items()))}


def _validate_partition(partitions: dict[str, list[dict[str, Any]]]) -> None:
    ids, pixels = set(), {}
    for split in SPLIT_NAMES:
        pixels[split] = set()
        for row in partitions[split]:
            _validate_source_row(row)
            if row.get("split") != split or row.get("category_normalized") != normalize_category(row["category_raw"]):
                raise ValueError(f"Invalid processed labels in {row['sample_id']}")
            if row["source_split"] != ("test" if split == "test" else "train"):
                raise ValueError("Upstream test data entered training/validation or vice versa")
            if row["sample_id"] in ids:
                raise ValueError("Sample ID overlaps prepared splits")
            ids.add(row["sample_id"])
            pixels[split].add(row["pixel_sha256"])
        if not partitions[split]:
            raise ValueError(f"Prepared {split} split is empty")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if pixels[first] & pixels[second]:
            raise ValueError(f"Oriented RGB pixels overlap {first} and {second}")


def prepare_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepare manifests from the current audited index without editing raw data.

    Test has priority: remove ALL training rows whose pixels occur in upstream
    test, then split the remaining image groups.  All captions and raw metadata
    are copied verbatim.  Excluded rows remain available in the source index.
    The manifest is published last, so interrupted writes fail hash validation
    instead of being accepted as a coherent preparation.
    """
    provenance = verify_raw(cfg)
    source, audit = _validated_source_records(cfg, provenance)
    reports, processed = _path(cfg, "reports"), _path(cfg, "processed")
    exclusion_report = _exclusions(source)
    excluded_ids = {row["sample_id"] for row in exclusion_report["exclusions"]}
    candidates = [row for row in source if row["source_split"] == "train" and row["sample_id"] not in excluded_ids]
    fraction, seed = cfg["data"]["validation_fraction"], cfg["data"]["split_seed"]
    val_groups, allocation = _stratified_validation_groups(candidates, fraction, seed)
    partitions: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for source_row in source:
        if source_row["sample_id"] in excluded_ids:
            continue
        split = "test" if source_row["source_split"] == "test" else (
            "val" if source_row["pixel_sha256"] in val_groups else "train")
        row = dict(source_row)
        row.update({"split": split, "category_normalized": normalize_category(row["category_raw"])})
        partitions[split].append(row)
    _validate_partition(partitions)
    if len(partitions["test"]) != sum(row["source_split"] == "test" for row in source):
        raise ValueError("Upstream test rows were lost")
    if sum(map(len, partitions.values())) + len(excluded_ids) != len(source):
        raise ValueError("Prepared partitions do not account for every source row")
    # This report is regenerated by both audit and prepare; stale manually
    # exported exclusion lists can never silently drive a different partition.
    write_json(reports / "train_exclusions.json", exclusion_report)
    split_info = {}
    for split in SPLIT_NAMES:
        target = processed / f"{split}.jsonl"
        write_jsonl(target, partitions[split])
        split_info[split] = {"path": target.name, "sha256": file_sha256(target),
                             **_split_statistics(partitions[split])}
    manifest = {
        "format_version": 1, "repo_id": REPO_ID, "revision": REVISION,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": seed, "validation_fraction": fraction,
        "actual_validation_row_fraction": len(partitions["val"]) / len(candidates),
        "source_rows": len(source), "eligible_train_rows": len(candidates),
        "excluded_train_rows": len(excluded_ids), "raw_test_preserved": True,
        "captions_preserved_verbatim": True, "pairwise_pixel_hash_overlap": 0,
        "category_normalization": "collapse Unicode whitespace, then str.casefold(); caption and category_raw unchanged",
        "leakage_policy": "Exclude all train pixels present in upstream test; allocate remaining complete pixel groups to train or val",
        "duplicate_detection_scope": audit["duplicate_detection_scope"],
        "raw_provenance": provenance,
        "inputs": {"source_index_sha256": file_sha256(reports / "source_index.jsonl"),
                   "data_audit_sha256": file_sha256(reports / "data_audit.json"),
                   "duplicate_groups_sha256": file_sha256(reports / "duplicate_groups.json"),
                   "train_exclusions_sha256": file_sha256(reports / "train_exclusions.json")},
        # A prepared split is published only after a provenance-bound audit;
        # the explicit flag also makes that requirement visible to readers of
        # this manifest without inspecting the implementation.
        "audit_embedded_source_index_hash": "source_index_sha256" in audit,
        "stratification": allocation, "splits": split_info,
    }
    write_json(processed / "splits.json", manifest)
    print("PREPARED " + ", ".join(f"{name}={len(partitions[name])}" for name in SPLIT_NAMES)
          + f"; excluded_train={len(excluded_ids)}", flush=True)
    return manifest


def _checked_manifests(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    processed, raw, reports = _path(cfg, "processed"), _path(cfg, "raw"), _path(cfg, "reports")
    manifest = _read_json(processed / "splits.json")
    if manifest.get("format_version") != 1 or manifest.get("repo_id") != REPO_ID or manifest.get("revision") != REVISION:
        raise ValueError("Prepared split manifest has an incompatible format or dataset revision")
    if manifest.get("split_seed") != cfg["data"]["split_seed"] or manifest.get("validation_fraction") != cfg["data"]["validation_fraction"]:
        raise ValueError("Split settings changed; explicitly run data.py prepare again")
    for key, filename in (("source_metadata_sha256", "source_metadata.json"),
                          ("download_manifest_sha256", "download_manifest.json")):
        if file_sha256(raw / filename) != manifest["raw_provenance"][key]:
            raise ValueError(f"Raw provenance file changed: {filename}")
    metadata = _read_json(raw / "source_metadata.json")
    _selected_files(metadata)
    for key, filename in (("source_index_sha256", "source_index.jsonl"),
                          ("data_audit_sha256", "data_audit.json"),
                          ("duplicate_groups_sha256", "duplicate_groups.json"),
                          ("train_exclusions_sha256", "train_exclusions.json")):
        if file_sha256(reports / filename) != manifest["inputs"][key]:
            raise ValueError(f"Preparation input changed: {filename}; explicitly rerun prepare")
    if set(manifest["splits"]) != set(SPLIT_NAMES):
        raise ValueError("Prepared manifest must describe train, val, and test")
    partitions = {}
    for split in SPLIT_NAMES:
        info = manifest["splits"][split]
        if info.get("path") != f"{split}.jsonl":
            raise ValueError("Unexpected split filename")
        target = _safe_child(processed, info["path"])
        if file_sha256(target) != info["sha256"]:
            raise ValueError(f"Prepared {split} manifest hash mismatch")
        partitions[split] = read_jsonl(target)
        if _split_statistics(partitions[split]) != {key: info[key] for key in ("rows", "pixel_groups", "categories_normalized", "category_counts")}:
            raise ValueError(f"Prepared {split} statistics disagree with splits.json")
    _validate_partition(partitions)
    return manifest, partitions


def load_split(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Load a split only after checking all manifest hashes and provenance links.

    Raw Parquet content hashes are checked by download/audit/verify/prepare.
    Rehashing gigabytes for every DataLoader construction would waste I/O, so
    this loader checks the pinned metadata/receipt and all small input/output
    manifests.  Each fetched image additionally checks its embedded byte hash.
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    _, partitions = _checked_manifests(cfg)
    return partitions[split]


def manifest_sha(cfg: dict[str, Any], split: str) -> str:
    """Return the checked SHA-256 of the particular split JSONL file."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")
    manifest, _ = _checked_manifests(cfg)
    return manifest["splits"][split]["sha256"]


def verify_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify raw upstream hashes and any existing audit/prepared artifacts."""
    provenance = verify_raw(cfg)
    result: dict[str, Any] = {"raw_files_verified": len(provenance["files"]), "revision": REVISION}
    reports = _path(cfg, "reports")
    if (reports / "source_index.jsonl").exists() or (reports / "data_audit.json").exists():
        records, _ = _validated_source_records(cfg, provenance)
        result["audited_rows"] = len(records)
        expected_exclusions = _exclusions(records)
        if (reports / "train_exclusions.json").exists():
            existing = _read_json(reports / "train_exclusions.json")
            if existing.get("dataset_revision") != REVISION or existing.get("exclusions") != expected_exclusions["exclusions"]:
                raise ValueError("Train exclusion list differs from the current source index")
    if (_path(cfg, "processed") / "splits.json").exists():
        manifest, partitions = _checked_manifests(cfg)
        if manifest["raw_provenance"] != provenance:
            raise ValueError("Prepared splits refer to different raw file hashes")
        result["prepared_rows"] = {split: len(rows) for split, rows in partitions.items()}
    print(json.dumps(result, indent=2), flush=True)
    return result


class ParquetImageStore:
    """Random image access with a bounded row-group LRU local to each worker.

    Reading a whole shard for every sample causes enormous repeated I/O and
    memory allocation.  Metadata maps a row to its containing row group; only
    that group's image column is read.  At most max_cached_row_groups tables
    remain resident per worker.  Forked or pickled DataLoader workers reopen
    their own files and never reuse a parent's Arrow file handles.
    """

    def __init__(self, raw_root: str | Path, max_cached_row_groups: int = 2):
        if isinstance(max_cached_row_groups, bool) or not isinstance(max_cached_row_groups, int) or max_cached_row_groups < 1:
            raise ValueError("max_cached_row_groups must be positive")
        self.raw_root = Path(raw_root).resolve()
        self.max_cached_row_groups = max_cached_row_groups
        self._pid: int | None = None
        self._files: dict[str, tuple[Any, list[int]]] = {}
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update({"_pid": None, "_files": {}, "_cache": OrderedDict()})
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pid = None
        self._files = {}
        self._cache = OrderedDict()

    def close(self) -> None:
        self._cache.clear()
        for parquet, _ in self._files.values():
            parquet.close()
        self._files.clear()

    def _worker_files(self) -> None:
        pid = os.getpid()
        if self._pid != pid:
            # A fork can inherit references even without invoking __getstate__.
            # Dropping them forces independent handles in the new process.
            self.close()
            self._pid = pid

    def _encoded(self, record: dict[str, Any]) -> bytes:
        import pyarrow.parquet as pq

        self._worker_files()
        shard, row = record["shard"], record["row_in_shard"]
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ValueError("Invalid Parquet row offset")
        if shard not in self._files:
            parquet = pq.ParquetFile(_safe_child(self.raw_root, shard))
            ends, total = [], 0
            for group in range(parquet.num_row_groups):
                total += parquet.metadata.row_group(group).num_rows
                ends.append(total)
            self._files[shard] = parquet, ends
        parquet, ends = self._files[shard]
        if not ends or row >= ends[-1]:
            raise IndexError(f"Parquet row is outside {shard}: {row}")
        group = bisect_right(ends, row)
        key = (shard, group)
        if key not in self._cache:
            # Arrow retains at most a small number of compressed-image columns,
            # never every dataset image or a materialized full-shard Python list.
            table = parquet.read_row_group(group, columns=["image"], use_threads=False)
            self._cache[key] = table
            while len(self._cache) > self.max_cached_row_groups:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        local_row = row - (ends[group - 1] if group else 0)
        image_field = self._cache[key].column("image")[local_row].as_py()
        encoded = image_field.get("bytes") if isinstance(image_field, dict) else None
        if not encoded:
            raise ValueError(f"Missing embedded image bytes for {record['sample_id']}")
        if hashlib.sha256(encoded).hexdigest() != record["image_sha256"]:
            raise ValueError(f"Image byte hash mismatch for {record['sample_id']}; check shard and row provenance")
        return encoded

    def get_image(self, record: dict[str, Any]) -> Any:
        """Return a fully decoded, detached, EXIF-oriented RGB PIL image."""
        from PIL import Image, ImageOps

        with Image.open(BytesIO(self._encoded(record))) as original:
            original.load()
            image = ImageOps.exif_transpose(original).convert("RGB")
        if image.size != (record["width"], record["height"]) or _pixel_hash(image) != record["pixel_sha256"]:
            raise ValueError(f"Oriented image pixels differ from the audit for {record['sample_id']}")
        return image


class FoodDataset:
    """Map-style PyTorch-compatible dataset with deterministic SDXL preprocessing.

    Framework imports occur in __getitem__, so download/audit/prepare do not
    require torch, torchvision, a GPU, or model weights.  Images use short-edge
    resize and center crop; there is no flip or caption dropout.  The unchanged
    full caption is tokenized later by the experiment's shared text code.
    """

    def __init__(self, cfg: dict[str, Any], records: list[dict[str, Any]], resolution: int):
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 8 or resolution % 8:
            raise ValueError("Training resolution must be a positive multiple of eight")
        self.records = records
        self.resolution = resolution
        self.store = ParquetImageStore(_path(cfg, "raw"), cfg["data"]["row_group_cache"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        record = self.records[index]
        image = self.store.get_image(record)
        original_width, original_height = image.size
        scale = self.resolution / min(original_width, original_height)
        resized_width = max(self.resolution, round(original_width * scale))
        resized_height = max(self.resolution, round(original_height * scale))
        image = image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)
        crop_x = (resized_width - self.resolution) // 2
        crop_y = (resized_height - self.resolution) // 2
        image = image.crop((crop_x, crop_y, crop_x + self.resolution, crop_y + self.resolution))
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).contiguous()
        # SDXL expects (original height, original width, crop top, crop left,
        # target height, target width); crop coordinates refer to resized pixels.
        time_ids = torch.tensor((original_height, original_width, crop_y, crop_x,
                                 self.resolution, self.resolution), dtype=torch.long)
        return {"pixel_values": pixels, "caption": record["caption"],
                "sample_id": record["sample_id"], "time_ids": time_ids}


def collate_training_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {"pixel_values": torch.stack([row["pixel_values"] for row in examples]),
            "captions": [row["caption"] for row in examples],
            "sample_ids": [row["sample_id"] for row in examples],
            "time_ids": torch.stack([row["time_ids"] for row in examples])}


class EpochSampler:
    """Stable epoch permutations without advancing Python or torch global RNGs.

    Accelerate may shard this sampler's batches across processes.  All ranks
    begin from the same seed and epoch; checkpoint resume can reconstruct the
    epoch order and skip already-consumed batches without changing noise RNGs.
    """

    def __init__(self, length: int, seed: int):
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("Sampler length must be a nonnegative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Sampler seed must be an integer")
        self.length, self.seed, self.epoch = length, seed, 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Sampler epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        order = list(range(self.length))
        random.Random(self.seed + self.epoch).shuffle(order)
        return iter(order)


def main() -> None:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    if len(sys.argv) == 1:
        # PyCharm's Run action starts the script without a subcommand. The
        # existing local dataset is the input, and the audit is refreshed before
        # prepare_splits reads its provenance. Resolve config beside the script
        # so PyCharm's choice of working directory does not affect this path.
        cfg = load_config(default_config)
        print("Auditing local Food500Cap data and preparing train/val/test splits...", flush=True)
        audit_dataset(cfg)
        prepare_splits(cfg)
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "audit", "verify", "prepare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, default=default_config)
        if command == "download":
            subparser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "download":
        download_dataset(cfg, args.workers)
    elif args.command == "audit":
        audit_dataset(cfg)
    elif args.command == "verify":
        verify_dataset(cfg)
    elif args.command == "prepare":
        prepare_splits(cfg)


if __name__ == "__main__":
    main()
