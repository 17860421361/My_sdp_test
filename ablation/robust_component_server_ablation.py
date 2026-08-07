"""Run the full-data, one-seed Robust phase-component ablation on a server.

This script is intentionally outside the WSDP source tree.  It fixes every
factor except the phase-calibration variant:

    savgol(w=7,p=3) -> IQR(1.5) -> PHASE VARIANT
    -> nearest15 -> z-score [amplitude, phase] -> length 1500 -> mlpmodel

The default protocol uses split seed 42 and model seed 42.  Gait trains for 60
epochs and Widar for 80 epochs, matching the current ``full_test_*_new.py``
experiments.  All variants reuse the same deterministic file order and the
same source ``_create_data_split`` indices.

The implementation is designed for the full Gait dataset.  The shared
Savgol+IQR prefix is cached once as one ``.npy`` file per sample.  Each phase
variant is then streamed into a single fixed-shape float32 memmap, avoiding the
old raw+prefix+calibrated+tail+split memory peak.  A completed condition can be
skipped safely; an interrupted condition keeps its processed memmap so model
training can be retried without repeating that condition's preprocessing.

Server examples (run from the project root)::

    CUDA_VISIBLE_DEVICES=0 python -u ablation/robust_component_server_ablation.py \
        --dataset gait --suite core --workers 4

    CUDA_VISIBLE_DEVICES=0 python -u ablation/robust_component_server_ablation.py \
        --dataset widar --suite core --workers 4

Use ``--dry-run`` first to verify paths without reading CSI payloads.  Use
``--preprocess-only --max-samples 18`` with a separate output directory for a
small server smoke test.  ``--max-samples 0`` (the default) means full data.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import socket
import sys
import tempfile
import time
import traceback
import types
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wsdp_mplconfig")
)

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "robust_component_server_results"
PROTOCOL_VERSION = "robust-components-server-v3"


def find_wsdp_src() -> Path:
    """Find WSDP in both the local outer layout and the server layout."""
    candidates = (
        PROJECT_ROOT
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        PROJECT_ROOT
        / "SDP"
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        PROJECT_ROOT / "src",
    )
    for candidate in candidates:
        if (candidate / "wsdp").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find WSDP source. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


WSDP_SRC = find_wsdp_src()
sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp import readers  # noqa: E402
from wsdp.algorithms import execute_pipeline  # noqa: E402
from wsdp.algorithms.amplitude import normalize_amplitude  # noqa: E402
from wsdp.algorithms.phase import robust_phase_sanitization  # noqa: E402
from wsdp.core import _create_data_split, _evaluate_model  # noqa: E402
from wsdp.models import create_model  # noqa: E402
from wsdp.processors.base_processor import (  # noqa: E402
    _parse_file_info_from_filename,
    _selector,
)
from wsdp.utils import load_params, train_model  # noqa: E402


PREFIX_STEPS = {
    "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
    "outliers": {"method": "iqr", "factor": 1.5},
}
INTERPOLATION_STEP = {"method": "nearest", "target_K": 15}
NORMALIZATION_METHOD = "z-score"
PADDING_LENGTH = 1500
PREFIX_RAW_SIZE_FACTOR = 4.0

ALL_CONDITIONS = (
    "linear_reference",
    "robust_first50",
    "no_calibration",
    "common_only",
    "detrend_first50_only",
    "robust_shared_first50",
    "robust_window_limited",
    "robust_fullspan50",
    "detrend_fullspan50_only",
)

# These eight conditions answer the root-cause question with the fewest full
# MLP trainings.  ``full`` adds one standalone full-visible-span control.
CORE_CONDITIONS = ALL_CONDITIONS[:8]
FULL_CONDITIONS = ALL_CONDITIONS

DATASET_DEFAULTS = {
    "gait": {"epochs": 60},
    "widar": {"epochs": 80},
}

SUMMARY_FIELDS = (
    "dataset",
    "condition",
    "status",
    "model",
    "model_seed",
    "split_seed",
    "epochs",
    "best_val_acc",
    "test_acc",
    "delta_vs_linear_pp",
    "delta_vs_robust_pp",
    "train_size",
    "val_size",
    "test_size",
    "input_shape",
    "parameter_count",
    "preprocess_duration_sec",
    "training_duration_sec",
    "total_duration_sec",
    "config_fingerprint",
    "checkpoint",
    "error",
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")


def acquire_run_lock(output_root: Path, dataset: str) -> tuple[Path, str]:
    """Exclusively lock one dataset below an output root.

    Gait and Widar may use the same output root concurrently, but two jobs for
    the same dataset would corrupt memmaps and checkpoints.  A lock left by a
    killed process is deliberately not removed automatically: the operator can
    inspect it before deleting it.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / f".{dataset}.lock"
    token = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
    payload = {
        "dataset": dataset,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "token": token,
    }
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        try:
            owner = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            owner = "<unreadable lock>"
        raise RuntimeError(
            f"Another {dataset} run appears active for {output_root}. "
            f"Lock: {lock_path}; owner: {owner}. If the recorded process is "
            "definitely gone, remove only this lock file and rerun."
        ) from error
    try:
        os.write(descriptor, canonical_json(payload).encode("utf-8"))
    except BaseException:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return lock_path, token


def release_run_lock(lock_path: Path, token: str) -> None:
    try:
        payload = read_json(lock_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if payload and payload.get("token") == token:
        lock_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_python_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_prefix_id(manifest_fingerprint: str, wsdp_hash: str) -> str:
    return fingerprint(
        {
            "dataset_manifest": manifest_fingerprint,
            "prefix_steps": PREFIX_STEPS,
            "wsdp_hash": wsdp_hash,
            "format": "per-sample-npy-v1",
        }
    )[:20]


def gib(byte_count: int | float) -> float:
    return float(byte_count) / 2**30


def check_disk_space(
    entries: list[dict[str, Any]],
    dataset_dir: Path,
    prefix_id: str,
    estimated_processed_bytes: int,
    pending_conditions: list[str],
    keep_processed: bool,
    minimum_free_gib: float,
    skip_check: bool,
) -> None:
    """Conservatively estimate peak disk needed before a long server run."""
    raw_bytes = sum(int(entry["source_size"]) for entry in entries)
    estimated_prefix_bytes = int(raw_bytes * PREFIX_RAW_SIZE_FACTOR)
    prefix_dir = dataset_dir / "cache" / f"prefix_{prefix_id}"
    cached_files = list((prefix_dir / "samples").glob("*.npy"))
    cached_prefix_bytes = sum(path.stat().st_size for path in cached_files)
    completion = read_json(prefix_dir / "completion.json")
    prefix_complete = bool(
        completion
        and completion.get("status") == "ok"
        and completion.get("prefix_id") == prefix_id
        and len(cached_files) == len(entries)
    )
    remaining_prefix_bytes = (
        0
        if prefix_complete
        else max(0, estimated_prefix_bytes - cached_prefix_bytes)
    )
    processed_copies = (
        len(pending_conditions)
        if keep_processed
        else min(1, len(pending_conditions))
    )
    safety_bytes = int(minimum_free_gib * 2**30)
    required_bytes = (
        remaining_prefix_bytes
        + processed_copies * estimated_processed_bytes
        + safety_bytes
    )
    free_bytes = shutil.disk_usage(dataset_dir.parent).free
    print(
        "Disk estimate: "
        f"raw={gib(raw_bytes):.2f} GiB, "
        f"prefix_total~{gib(estimated_prefix_bytes):.2f} GiB "
        f"({gib(cached_prefix_bytes):.2f} GiB cached), "
        f"processed_peak~{gib(processed_copies * estimated_processed_bytes):.2f} GiB, "
        f"free={gib(free_bytes):.2f} GiB, safety={minimum_free_gib:.2f} GiB",
        flush=True,
    )
    if free_bytes < required_bytes:
        message = (
            f"Estimated free-space requirement is {gib(required_bytes):.2f} GiB, "
            f"but only {gib(free_bytes):.2f} GiB is free below "
            f"{dataset_dir.parent}. Choose a larger --output-root, delete old "
            "ablation caches, or explicitly pass --skip-disk-check after "
            "verifying capacity yourself."
        )
        if skip_check:
            print(f"WARNING: {message}", flush=True)
        else:
            raise RuntimeError(message)


def resolve_data_path(dataset: str, explicit: str | None) -> Path:
    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not result.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {result}")
        return result

    roots = (PROJECT_ROOT, PROJECT_ROOT / "SDP", PROJECT_ROOT.parent)
    relative_candidates = {
        "gait": (
            Path("sdp_dataset") / "Gait_Dataset" / "CSI_Gait",
            Path("sdp_dataset") / "Gait_Dataset",
            Path("sdp_dataset") / "gait",
        ),
        "widar": (
            Path("sdp_dataset") / "widar_common3",
            Path("sdp_dataset") / "widar",
        ),
    }[dataset]
    candidates = tuple(root / relative for root in roots for relative in relative_candidates)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find {dataset} data. Checked: "
        + ", ".join(str(path) for path in candidates)
        + ". Pass --data-path explicitly if the server uses another layout."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def raw_metadata_for_path(path: Path, dataset: str) -> tuple[Any, Any] | None:
    parsed = _parse_file_info_from_filename(path.name, dataset)
    if parsed is None:
        return None
    return _selector(parsed, dataset)


def select_balanced_entries(
    entries: list[dict[str, Any]], max_samples: int, seed: int
) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(entries) <= max_samples:
        return entries
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        buckets[int(entry["raw_label"])].append(entry)
    rng = np.random.default_rng(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_samples:
        progressed = False
        for label in sorted(buckets):
            if buckets[label] and len(selected) < max_samples:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: item["relative_path"])


def discover_entries(
    data_path: Path, dataset: str, max_samples: int, seed: int
) -> tuple[list[dict[str, Any]], str, int]:
    all_files = sorted(
        (
            path
            for path in data_path.rglob("*")
            if path.is_file() and "truth" not in path.name
        ),
        key=lambda item: item.relative_to(data_path).as_posix(),
    )
    entries: list[dict[str, Any]] = []
    for path in all_files:
        metadata = raw_metadata_for_path(path, dataset)
        if metadata is None:
            continue
        label, group = metadata
        stat = path.stat()
        entries.append(
            {
                "source_path": str(path.resolve()),
                "relative_path": path.relative_to(data_path).as_posix(),
                "source_size": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                "raw_label": str(label),
                "raw_group": str(group),
            }
        )
    if not entries:
        raise RuntimeError(f"No parseable {dataset} files found below {data_path}")
    entries = select_balanced_entries(entries, max_samples, seed)
    manifest_payload = [
        {
            key: entry[key]
            for key in (
                "relative_path",
                "source_size",
                "source_mtime_ns",
                "raw_label",
                "raw_group",
            )
        }
        for entry in entries
    ]
    return entries, fingerprint(manifest_payload), len(all_files)


def encode_targets(
    entries: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    # Widar/Gait selectors return integers.  Preserve their numeric ordering:
    # GroupShuffleSplit seed 42 can choose different held-out groups if values
    # such as 2 and 10 are first reordered lexicographically as strings.
    raw_labels = [int(entry["raw_label"]) for entry in entries]
    raw_groups = [int(entry["raw_group"]) for entry in entries]
    unique_labels = sorted(set(raw_labels))
    unique_groups = sorted(set(raw_groups))
    label_map = {value: index for index, value in enumerate(unique_labels)}
    group_map = {value: index for index, value in enumerate(unique_groups)}
    labels = np.asarray([label_map[value] for value in raw_labels], dtype=np.int64)
    groups = np.asarray([group_map[value] for value in raw_groups], dtype=np.int64)
    return (
        labels,
        groups,
        [str(value) for value in unique_labels],
        [str(value) for value in unique_groups],
    )


def parse_csi_item(item: Any, dataset: str) -> tuple[np.ndarray, str, str]:
    parsed = _parse_file_info_from_filename(item.file_name, dataset)
    if parsed is None:
        raise ValueError(f"Cannot parse label/group from {item.file_name!r}")
    label, group = _selector(parsed, dataset)
    frames = sorted(item.frames, key=lambda frame: frame.timestamp)
    tensors = [frame.csi_array for frame in frames]
    if not tensors:
        raise ValueError(f"No CSI frames in {item.file_name!r}")
    csi = np.stack(tensors, axis=0)
    if csi.ndim == 2:
        csi = csi[..., None]
    if csi.ndim != 3 or csi.shape[0] < 2 or not np.iscomplexobj(csi):
        raise ValueError(
            f"Expected complex (T,F,A) with T>=2, got {csi.shape} {csi.dtype}"
        )
    return csi, str(label), str(group)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def prefix_file_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    index, source_text, target_text, dataset, expected_label, expected_group = task
    source = Path(source_text)
    target = Path(target_text)
    reader = readers.get_reader_class(dataset)()
    if not reader.sniff(str(source)):
        raise ValueError(f"Reader format mismatch: {source}")
    loaded = reader.read_file(str(source))
    items = loaded if isinstance(loaded, list) else [loaded]
    if len(items) != 1:
        raise ValueError(
            f"Expected one CSI sample per file, got {len(items)} from {source}"
        )
    csi, label, group = parse_csi_item(items[0], dataset)
    if label != expected_label or group != expected_group:
        raise RuntimeError(
            f"Filename metadata changed after reading {source}: "
            f"expected ({expected_label},{expected_group}), got ({label},{group})"
        )
    prefixed = execute_pipeline(csi, PREFIX_STEPS, dataset=dataset)
    if prefixed.shape != csi.shape or not np.iscomplexobj(prefixed):
        raise RuntimeError(
            f"Prefix produced invalid output for {source}: {prefixed.shape} {prefixed.dtype}"
        )
    save_npy_atomic(target, prefixed)
    return {
        "index": int(index),
        "frames": int(prefixed.shape[0]),
        "shape": list(prefixed.shape),
        "dtype": str(prefixed.dtype),
    }


def bounded_map(
    function: Any, tasks: list[Any], workers: int
) -> Iterator[Any]:
    """Map with bounded in-flight futures and unordered, indexed results.

    ``Executor.map`` submits the full iterable eagerly on supported Python
    versions.  A slow early CSI file can then leave thousands of completed
    0.5-MiB model inputs retained by Future objects.  Every worker result here
    carries its sample index, so ordering is unnecessary and a small bounded
    queue keeps memory independent of dataset size.
    """
    if workers == 1:
        yield from map(function, tasks)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        task_iterator = iter(tasks)
        in_flight = set()
        for _ in range(min(len(tasks), max(workers * 2, 1))):
            try:
                in_flight.add(executor.submit(function, next(task_iterator)))
            except StopIteration:
                break
        while in_flight:
            completed, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    in_flight.add(executor.submit(function, next(task_iterator)))
                except StopIteration:
                    pass


def valid_prefix_cache(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return value.ndim == 3 and value.shape[0] >= 2 and np.iscomplexobj(value)
    except Exception:
        return False


def build_prefix_cache(
    entries: list[dict[str, Any]],
    dataset: str,
    dataset_dir: Path,
    manifest_fingerprint: str,
    wsdp_hash: str,
    workers: int,
) -> tuple[list[Path], list[dict[str, Any]], str]:
    prefix_id = make_prefix_id(manifest_fingerprint, wsdp_hash)
    cache_dir = dataset_dir / "cache" / f"prefix_{prefix_id}"
    sample_dir = cache_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [sample_dir / f"{index:06d}.npy" for index in range(len(entries))]
    completion_path = cache_dir / "completion.json"
    completion = read_json(completion_path)

    metadata_by_index: dict[int, dict[str, Any]] = {}
    if completion and completion.get("prefix_id") == prefix_id:
        if all(valid_prefix_cache(path) for path in cache_paths):
            print(f"Reuse complete prefix cache: {cache_dir}", flush=True)
        else:
            completion = None

    if completion is None:
        pending_tasks = []
        for index, (entry, target) in enumerate(zip(entries, cache_paths)):
            if valid_prefix_cache(target):
                value = np.load(target, mmap_mode="r", allow_pickle=False)
                metadata_by_index[index] = {
                    "index": index,
                    "frames": int(value.shape[0]),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                continue
            pending_tasks.append(
                (
                    index,
                    entry["source_path"],
                    str(target),
                    dataset,
                    entry["raw_label"],
                    entry["raw_group"],
                )
            )
        print(
            f"Prefix cache {prefix_id}: {len(entries) - len(pending_tasks)} cached, "
            f"{len(pending_tasks)} pending",
            flush=True,
        )
        for completed_count, result in enumerate(
            bounded_map(prefix_file_worker, pending_tasks, workers), 1
        ):
            metadata_by_index[int(result["index"])] = result
            if completed_count % 100 == 0 or completed_count == len(pending_tasks):
                print(
                    f"  prefix files: {completed_count}/{len(pending_tasks)}",
                    flush=True,
                )
        if not all(valid_prefix_cache(path) for path in cache_paths):
            raise RuntimeError("Prefix cache did not produce every expected sample")
        atomic_write_json(
            completion_path,
            {
                "status": "ok",
                "prefix_id": prefix_id,
                "sample_count": len(entries),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    rows: list[dict[str, Any]] = []
    for index, (entry, cache_path) in enumerate(zip(entries, cache_paths)):
        metadata = metadata_by_index.get(index)
        if metadata is None:
            value = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            metadata = {
                "frames": int(value.shape[0]),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        rows.append(
            {
                "sample_index": index,
                "source_relative_path": entry["relative_path"],
                "source_size": entry["source_size"],
                "source_mtime_ns": entry["source_mtime_ns"],
                "raw_label": entry["raw_label"],
                "raw_group": entry["raw_group"],
                "prefix_cache_relative_path": cache_path.relative_to(dataset_dir).as_posix(),
                "frames": metadata["frames"],
                "shape": json.dumps(metadata["shape"]),
                "dtype": metadata["dtype"],
            }
        )
    atomic_write_csv(cache_dir / "sample_manifest.csv", rows, rows[0].keys())
    return cache_paths, rows, prefix_id


def theil_sen_slope(phases: np.ndarray, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        return np.zeros(phases.shape[1:], dtype=np.float64)
    left, right = np.triu_indices(indices.size, k=1)
    left_indices = indices[left]
    right_indices = indices[right]
    denominators = (right_indices - left_indices).astype(np.float64)[:, None, None]
    slopes = (phases[right_indices] - phases[left_indices]) / denominators
    return np.median(slopes, axis=0)


def unwrap_time_float64(csi: np.ndarray) -> np.ndarray:
    # The cast after unwrap mirrors the source's float64 destination array.
    return np.unwrap(np.angle(csi), axis=0).astype(np.float64, copy=False)


def reconstruct_like(csi: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return (np.abs(csi) * np.exp(1j * phase)).astype(csi.dtype, copy=False)


def robust_first50_mirror(csi: np.ndarray) -> np.ndarray:
    phase = unwrap_time_float64(csi)
    centered = phase - np.median(phase, axis=1, keepdims=True)
    if csi.shape[0] < 3:
        return reconstruct_like(csi, centered)
    fit_count = min(csi.shape[0], 50)
    slope = theil_sen_slope(centered, np.arange(fit_count, dtype=np.int64))
    times = np.arange(csi.shape[0], dtype=np.float64)[:, None, None]
    return reconstruct_like(csi, centered - times * slope[None, :, :])


def apply_phase_variant(csi: np.ndarray, condition: str, dataset: str) -> np.ndarray:
    if condition == "no_calibration":
        return np.asarray(csi).copy()
    if condition == "linear_reference":
        # Use the current source route, including its current tone-mapping behavior.
        return execute_pipeline(
            csi, {"calibrate": {"method": "linear"}}, dataset=dataset
        )
    if condition == "robust_first50":
        # The target condition is the source itself, never a custom approximation.
        return robust_phase_sanitization(csi)

    phase = unwrap_time_float64(csi)
    common = np.median(phase, axis=1, keepdims=True)
    centered = phase - common
    fit_count = min(csi.shape[0], 50)
    first_indices = np.arange(fit_count, dtype=np.int64)
    visible_frames = min(csi.shape[0], PADDING_LENGTH)
    visible_span_indices = np.unique(
        np.linspace(0, visible_frames - 1, fit_count, dtype=np.int64)
    )
    times = np.arange(csi.shape[0], dtype=np.float64)[:, None, None]

    if csi.shape[0] < 3:
        if condition in {
            "common_only",
            "robust_shared_first50",
            "robust_window_limited",
            "robust_fullspan50",
        }:
            corrected = centered
        elif condition in {"detrend_first50_only", "detrend_fullspan50_only"}:
            corrected = phase
        else:
            raise ValueError(f"Unknown condition: {condition}")
        return reconstruct_like(csi, corrected)

    if condition == "common_only":
        corrected = centered
    elif condition == "detrend_first50_only":
        slope = theil_sen_slope(phase, first_indices)
        corrected = phase - times * slope[None, :, :]
    elif condition == "detrend_fullspan50_only":
        slope = theil_sen_slope(phase, visible_span_indices)
        corrected = phase - times * slope[None, :, :]
    elif condition in {"robust_shared_first50", "robust_window_limited"}:
        slope = theil_sen_slope(centered, first_indices)
        if condition == "robust_shared_first50":
            shared = np.median(slope, axis=0, keepdims=True)
            corrected = centered - times * shared[None, :, :]
        else:
            limited_times = np.minimum(times, max(fit_count - 1, 0))
            corrected = centered - limited_times * slope[None, :, :]
    elif condition == "robust_fullspan50":
        slope = theil_sen_slope(centered, visible_span_indices)
        corrected = centered - times * slope[None, :, :]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return reconstruct_like(csi, corrected)


def resize_one(sample: np.ndarray, target_length: int) -> np.ndarray:
    if sample.shape[0] > target_length:
        return sample[:target_length]
    if sample.shape[0] < target_length:
        return np.pad(
            sample,
            ((0, target_length - sample.shape[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    return sample


def derive_nearest_indices(dataset: str, carriers: int, target_k: int) -> np.ndarray:
    marker = np.arange(carriers, dtype=np.float64)[None, :, None].astype(np.complex128)
    source = execute_pipeline(
        marker,
        {"interpolate": {"method": "nearest", "target_K": target_k}},
        dataset=dataset,
    )
    values = np.real(source[0, :, 0])
    indices = np.rint(values).astype(np.int64)
    if (
        indices.shape != (target_k,)
        or np.any(indices < 0)
        or np.any(indices >= carriers)
        or not np.allclose(values, indices, atol=1e-12, rtol=0.0)
    ):
        raise RuntimeError(f"Could not derive exact nearest indices: {values}")
    return indices


def explicit_tail(
    calibrated: np.ndarray,
    nearest_indices: np.ndarray,
    target_length: int,
) -> np.ndarray:
    interpolated = calibrated[:, nearest_indices, :]
    explicit = normalize_amplitude(
        interpolated,
        method=NORMALIZATION_METHOD,
        return_phase_channels=True,
    )
    fixed = resize_one(explicit, target_length).astype(np.float32, copy=False)
    if np.iscomplexobj(fixed) or not np.all(np.isfinite(fixed)):
        raise RuntimeError("Tail produced complex or non-finite model input")
    return fixed


def adjacent_phase(csi: np.ndarray) -> np.ndarray:
    return np.angle(csi[:, 1:, :] * np.conj(csi[:, :-1, :]))


def diagnostic_metrics(
    source: np.ndarray, result: np.ndarray, condition: str, dataset: str
) -> dict[str, float | str]:
    baseline = apply_phase_variant(source, "common_only", dataset)
    source_jump = np.abs(adjacent_phase(source))
    result_jump = np.abs(adjacent_phase(result))
    baseline_relative = adjacent_phase(baseline)
    result_relative = adjacent_phase(result)
    relative_rotation = np.abs(
        np.angle(np.exp(1j * (result_relative - baseline_relative)))
    )
    quarter = max(1, result.shape[0] // 4)
    horizon = min(PADDING_LENGTH - 1, result.shape[0] - 1)
    horizon_rotation = relative_rotation[horizon]
    return {
        "condition": condition,
        "max_amplitude_error": float(
            np.max(np.abs(np.abs(result) - np.abs(source)))
        ),
        "source_adjacent_jump_mean_rad": float(np.mean(source_jump)),
        "result_adjacent_jump_early_mean_rad": float(np.mean(result_jump[:quarter])),
        "result_adjacent_jump_late_mean_rad": float(np.mean(result_jump[-quarter:])),
        "horizon_relative_rotation_mean_rad": float(np.mean(horizon_rotation)),
        "horizon_relative_rotation_gt_pi_over_2": float(
            np.mean(horizon_rotation > (np.pi / 2.0))
        ),
    }


def condition_worker(task: tuple[Any, ...]) -> tuple[int, np.ndarray, dict[str, Any] | None]:
    (
        index,
        prefix_path_text,
        condition,
        dataset,
        nearest_indices_list,
        target_length,
        collect_diagnostic,
    ) = task
    source = np.load(prefix_path_text, mmap_mode="r", allow_pickle=False)
    result = apply_phase_variant(source, condition, dataset)
    metrics = (
        diagnostic_metrics(source, result, condition, dataset)
        if collect_diagnostic
        else None
    )
    fixed = explicit_tail(
        result, np.asarray(nearest_indices_list, dtype=np.int64), target_length
    )
    return int(index), fixed, metrics


def summarize_numeric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    result: dict[str, Any] = {"samples": len(rows)}
    keys = sorted(set().union(*(row.keys() for row in rows)) - {"condition"})
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def run_equivalence_checks(
    prefix_paths: list[Path], dataset: str, dataset_dir: Path
) -> np.ndarray:
    sample = np.load(prefix_paths[0], mmap_mode="r", allow_pickle=False)
    mirror = robust_first50_mirror(sample)
    source_robust = robust_phase_sanitization(sample)
    robust_error = np.abs(
        mirror.astype(np.complex128) - source_robust.astype(np.complex128)
    )
    nearest_indices = derive_nearest_indices(dataset, sample.shape[1], 15)
    fast_nearest = sample[:, nearest_indices, :]
    source_nearest = execute_pipeline(
        sample,
        {"interpolate": INTERPOLATION_STEP},
        dataset=dataset,
    )
    nearest_error = np.abs(
        fast_nearest.astype(np.complex128) - source_nearest.astype(np.complex128)
    )
    fast_tail = explicit_tail(source_robust, nearest_indices, PADDING_LENGTH)
    source_tail = normalize_amplitude(
        execute_pipeline(
            source_robust,
            {"interpolate": INTERPOLATION_STEP},
            dataset=dataset,
        ),
        method="z-score",
        return_phase_channels=True,
    )
    source_tail = resize_one(source_tail, PADDING_LENGTH).astype(np.float32, copy=False)
    tail_error = np.abs(fast_tail.astype(np.float64) - source_tail.astype(np.float64))
    payload = {
        "robust_mirror_max_complex_absolute_error": float(np.max(robust_error)),
        "robust_mirror_allclose": bool(
            np.allclose(mirror, source_robust, rtol=1e-7, atol=1e-8)
        ),
        "derived_nearest_indices": nearest_indices.tolist(),
        "nearest_fast_path_max_complex_absolute_error": float(np.max(nearest_error)),
        "nearest_fast_path_allclose": bool(
            np.allclose(fast_nearest, source_nearest, rtol=0.0, atol=0.0)
        ),
        "full_tail_max_absolute_error": float(np.max(tail_error)),
        "full_tail_allclose": bool(
            np.allclose(fast_tail, source_tail, rtol=0.0, atol=0.0)
        ),
    }
    atomic_write_json(dataset_dir / "source_equivalence.json", payload)
    if not all(
        payload[key]
        for key in (
            "robust_mirror_allclose",
            "nearest_fast_path_allclose",
            "full_tail_allclose",
        )
    ):
        raise RuntimeError(
            "Source-equivalence gate failed; see "
            f"{dataset_dir / 'source_equivalence.json'}"
        )
    return nearest_indices


def build_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    dataset: str,
    split_seed: int,
    dataset_dir: Path,
    config_fingerprint: str,
    require_class_coverage: bool,
) -> dict[str, np.ndarray]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if require_class_coverage and len(np.unique(groups)) < 3:
        raise RuntimeError(
            "A formal run requires at least three groups; refusing to fall back "
            "to a random sample split. Check --data-path and dataset completeness."
        )
    index_payload = np.arange(len(labels), dtype=np.int64)[:, None]
    split = _create_data_split(
        index_payload,
        labels,
        groups,
        test_split=0.3,
        val_split=0.5,
        seed=split_seed,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=dataset,
        pipeline_steps={
            **PREFIX_STEPS,
            "normalize": {"method": "z-score"},
            "interpolate": INTERPOLATION_STEP,
        },
    )
    train_indices = np.asarray(split[0]).reshape(-1).astype(np.int64)
    val_indices = np.asarray(split[1]).reshape(-1).astype(np.int64)
    test_indices = np.asarray(split[2]).reshape(-1).astype(np.int64)
    result = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    if sum(len(values) for values in result.values()) != len(labels):
        raise RuntimeError("Split does not cover every sample exactly once")
    concatenated = np.concatenate(list(result.values()))
    if len(np.unique(concatenated)) != len(labels):
        raise RuntimeError("Split indices overlap or contain duplicates")
    if len(np.unique(groups)) >= 3:
        group_sets = {name: set(groups[indices].tolist()) for name, indices in result.items()}
        if (
            group_sets["train"] & group_sets["val"]
            or group_sets["train"] & group_sets["test"]
            or group_sets["val"] & group_sets["test"]
        ):
            raise RuntimeError("Group leakage detected across train/val/test")

    expected_classes = set(np.unique(labels).tolist())
    class_coverage = {
        name: set(np.unique(labels[indices]).tolist()) == expected_classes
        for name, indices in result.items()
    }
    if require_class_coverage and not all(class_coverage.values()):
        missing = {
            name: sorted(expected_classes - set(np.unique(labels[indices]).tolist()))
            for name, indices in result.items()
            if not class_coverage[name]
        }
        raise RuntimeError(
            "Formal split is missing classes: "
            f"{missing}. Check the server data path and full dataset copy."
        )
    if not all(class_coverage.values()):
        print(
            f"WARNING: smoke-test split does not contain every class: {class_coverage}",
            flush=True,
        )

    split_path = dataset_dir / "split_indices.npz"
    if split_path.exists():
        existing = np.load(split_path, allow_pickle=False)
        for name, values in result.items():
            if name not in existing or not np.array_equal(existing[name], values):
                raise RuntimeError(
                    "Existing split_indices.npz differs from this run. "
                    "Use a new --output-root rather than mixing protocols."
                )
    else:
        temporary = split_path.with_name(split_path.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **result)
        os.replace(temporary, split_path)

    def label_counts(indices: np.ndarray) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in Counter(labels[indices].tolist()).items()
        }

    metadata = {
        "config_fingerprint": config_fingerprint,
        "split_seed": split_seed,
        "source_function": "wsdp.core._create_data_split on stable sample indices",
        "sizes": {name: len(values) for name, values in result.items()},
        "label_counts": {name: label_counts(values) for name, values in result.items()},
        "all_classes_present": class_coverage,
        "groups": {
            name: sorted(set(groups[values].tolist())) for name, values in result.items()
        },
        "group_disjoint": len(np.unique(groups)) < 3
        or not (
            set(groups[train_indices]) & set(groups[val_indices])
            or set(groups[train_indices]) & set(groups[test_indices])
            or set(groups[val_indices]) & set(groups[test_indices])
        ),
    }
    atomic_write_json(dataset_dir / "split_metadata.json", metadata)
    return result


def build_processed_memmap(
    prefix_paths: list[Path],
    condition: str,
    dataset: str,
    condition_dir: Path,
    nearest_indices: np.ndarray,
    workers: int,
    diagnostic_samples: int,
    config_fingerprint: str,
) -> tuple[np.memmap, float]:
    cache_dir = condition_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_path = cache_dir / "processed.npy"
    done_path = cache_dir / "processed_done.npy"
    completion_path = cache_dir / "processed_completion.json"
    progress_path = cache_dir / "processed_progress.json"
    diagnostics_partial_path = cache_dir / "diagnostics_partial.json"
    completion = read_json(completion_path)
    first = np.load(prefix_paths[0], mmap_mode="r", allow_pickle=False)
    expected_shape = (
        len(prefix_paths),
        PADDING_LENGTH,
        len(nearest_indices),
        2 * first.shape[2],
    )
    if (
        completion
        and completion.get("status") == "ok"
        and completion.get("config_fingerprint") == config_fingerprint
        and completion.get("condition") == condition
        and processed_path.exists()
        and (condition_dir / "diagnostics.json").is_file()
    ):
        try:
            cached = np.load(processed_path, mmap_mode="r+", allow_pickle=False)
        except (OSError, ValueError):
            cached = None
        if (
            cached is not None
            and cached.shape == expected_shape
            and cached.dtype == np.float32
        ):
            print(f"Reuse processed cache: {processed_path}", flush=True)
            return cached, float(completion.get("duration_sec", 0.0))

    started = time.time()
    progress = read_json(progress_path)
    resume_matches = bool(
        progress
        and progress.get("condition") == condition
        and progress.get("config_fingerprint") == config_fingerprint
        and progress.get("shape") == list(expected_shape)
        and progress.get("dtype") == "float32"
        and progress.get("diagnostic_samples") == diagnostic_samples
        and processed_path.is_file()
        and done_path.is_file()
    )
    processed: np.memmap
    done: np.memmap
    if resume_matches:
        try:
            processed = np.load(processed_path, mmap_mode="r+", allow_pickle=False)
            done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
            resume_matches = bool(
                processed.shape == expected_shape
                and processed.dtype == np.float32
                and done.shape == (len(prefix_paths),)
                and done.dtype == np.bool_
            )
        except (OSError, ValueError):
            resume_matches = False

    diagnostic_by_index: dict[int, dict[str, Any]] = {}
    prior_duration = 0.0
    if resume_matches:
        prior_duration = float(progress.get("duration_sec", 0.0))
        partial = read_json(diagnostics_partial_path)
        if (
            partial
            and partial.get("condition") == condition
            and partial.get("config_fingerprint") == config_fingerprint
        ):
            for key, value in partial.get("per_sample", {}).items():
                diagnostic_by_index[int(key)] = value
        # A completed model input without its diagnostic row is safe to redo.
        # This keeps diagnostics complete even if the previous process stopped
        # between flushing the data bitmap and writing its JSON sidecar.
        for index in range(min(diagnostic_samples, len(prefix_paths))):
            if index not in diagnostic_by_index and bool(done[index]):
                done[index] = False
        done.flush()
    else:
        processed = np.lib.format.open_memmap(
            processed_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        done = np.lib.format.open_memmap(
            done_path,
            mode="w+",
            dtype=np.bool_,
            shape=(len(prefix_paths),),
        )
        done[:] = False
        done.flush()

    pending_indices = np.flatnonzero(~np.asarray(done)).tolist()
    completed_total = len(prefix_paths) - len(pending_indices)
    print(
        f"Processed cache {condition}: {completed_total} durable, "
        f"{len(pending_indices)} pending",
        flush=True,
    )
    atomic_write_json(
        progress_path,
        {
            "status": "in_progress",
            "condition": condition,
            "completed": completed_total,
            "total": len(prefix_paths),
            "shape": list(expected_shape),
            "dtype": "float32",
            "diagnostic_samples": diagnostic_samples,
            "duration_sec": prior_duration,
            "config_fingerprint": config_fingerprint,
        },
    )
    tasks = [
        (
            index,
            str(prefix_paths[index]),
            condition,
            dataset,
            nearest_indices.tolist(),
            PADDING_LENGTH,
            index < diagnostic_samples,
        )
        for index in pending_indices
    ]
    uncommitted_indices: list[int] = []
    for completed_count, (index, fixed, metrics) in enumerate(
        bounded_map(condition_worker, tasks, workers), 1
    ):
        if fixed.shape != expected_shape[1:]:
            raise RuntimeError(
                f"Unexpected model input for sample {index}: {fixed.shape}, "
                f"expected {expected_shape[1:]}"
            )
        processed[index] = fixed
        uncommitted_indices.append(index)
        if metrics is not None:
            metrics["sample_index"] = int(index)
            diagnostic_by_index[int(index)] = metrics
        if completed_count % 100 == 0 or completed_count == len(tasks):
            # Flush model inputs first, then mark their indices durable.  A
            # crash before the bitmap flush merely causes safe recomputation.
            processed.flush()
            done[uncommitted_indices] = True
            done.flush()
            uncommitted_indices.clear()
            completed_total = int(np.count_nonzero(done))
            duration = prior_duration + time.time() - started
            atomic_write_json(
                diagnostics_partial_path,
                {
                    "condition": condition,
                    "config_fingerprint": config_fingerprint,
                    "per_sample": {
                        str(key): diagnostic_by_index[key]
                        for key in sorted(diagnostic_by_index)
                    },
                },
            )
            atomic_write_json(
                progress_path,
                {
                    "status": "in_progress",
                    "condition": condition,
                    "completed": completed_total,
                    "total": len(prefix_paths),
                    "shape": list(expected_shape),
                    "dtype": "float32",
                    "diagnostic_samples": diagnostic_samples,
                    "duration_sec": duration,
                    "config_fingerprint": config_fingerprint,
                },
            )
            print(
                f"  {condition} model inputs: "
                f"{completed_total}/{len(prefix_paths)}",
                flush=True,
            )
    processed.flush()
    done.flush()
    if not bool(np.all(done)):
        missing = np.flatnonzero(~np.asarray(done))[:10].tolist()
        raise RuntimeError(f"Processed cache is incomplete; missing indices {missing}")
    diagnostic_rows = [
        diagnostic_by_index[index]
        for index in sorted(diagnostic_by_index)
        if index < diagnostic_samples
    ]
    if len(diagnostic_rows) != diagnostic_samples:
        raise RuntimeError(
            f"Expected {diagnostic_samples} diagnostic rows, got "
            f"{len(diagnostic_rows)}"
        )
    duration = prior_duration + time.time() - started
    atomic_write_json(
        condition_dir / "diagnostics.json",
        {
            "condition": condition,
            "aggregate": summarize_numeric_rows(diagnostic_rows),
            "per_sample": diagnostic_rows,
        },
    )
    atomic_write_json(
        completion_path,
        {
            "status": "ok",
            "condition": condition,
            "config_fingerprint": config_fingerprint,
            "shape": list(expected_shape),
            "dtype": "float32",
            "duration_sec": duration,
        },
    )
    atomic_write_json(
        progress_path,
        {
            "status": "ok",
            "condition": condition,
            "completed": len(prefix_paths),
            "total": len(prefix_paths),
            "shape": list(expected_shape),
            "dtype": "float32",
            "diagnostic_samples": diagnostic_samples,
            "duration_sec": duration,
            "config_fingerprint": config_fingerprint,
        },
    )
    return processed, duration


def resolve_device(device_text: str, allow_auto_cpu: bool = True) -> torch.device:
    if device_text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if not allow_auto_cpu:
            raise RuntimeError(
                "--device auto found no CUDA device for a formal full-data run. "
                "Request a GPU, pass --device cpu explicitly, or use --allow-cpu "
                "if the slow CPU run is intentional."
            )
        return torch.device("cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but torch.cuda.is_available() is false")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) "
            "are visible"
        )
    return device


def save_history_csv(history: dict[str, list[Any]], path: Path) -> None:
    fields = list(history)
    length = max((len(history[field]) for field in fields), default=0)
    rows = [
        {
            field: history[field][index] if index < len(history[field]) else ""
            for field in fields
        }
        for index in range(length)
    ]
    atomic_write_csv(path, rows, fields)


def torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_loaders(
    processed: np.ndarray,
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Explicit z-score [amplitude, phase] is already the final representation.
    # TensorDataset therefore performs no additional abs/angle conversion.
    full_dataset = TensorDataset(
        torch.from_numpy(processed), torch.from_numpy(labels).long()
    )

    def loader(name: str, shuffle: bool) -> DataLoader:
        subset = Subset(full_dataset, split_indices[name].tolist())
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    return loader("train", True), loader("val", False), loader("test", False)


def train_condition(
    processed: np.ndarray,
    labels: np.ndarray,
    unique_labels: list[str],
    split_indices: dict[str, np.ndarray],
    dataset: str,
    condition: str,
    condition_dir: Path,
    model_name: str,
    model_seed: int,
    split_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    config_fingerprint: str,
    preprocess_duration: float,
) -> dict[str, Any]:
    run_dir = condition_dir / f"seed_{model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pth"
    history_path = run_dir / "training_history.csv"
    predictions_path = run_dir / "test_predictions.npz"
    status_path = run_dir / "status.json"
    log_path = run_dir / "train_process.txt"
    for stale in (checkpoint_path, history_path, predictions_path):
        if stale.exists():
            stale.unlink()

    started = time.time()
    set_seed(model_seed)
    loaders = build_loaders(processed, labels, split_indices, batch_size)
    input_shape = tuple(processed.shape[1:])
    model = create_model(
        model_name, num_classes=len(unique_labels), input_shape=input_shape
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    running = {
        "status": "running",
        "stage": "training",
        "dataset": dataset,
        "condition": condition,
        "model": model_name,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "config_fingerprint": config_fingerprint,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(status_path, running)
    try:
        with log_path.open("a", encoding="utf-8") as log_handle:
            tee_out = Tee(sys.stdout, log_handle)
            tee_err = Tee(sys.stderr, log_handle)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                print("=" * 80)
                print(
                    f"dataset={dataset} condition={condition} model={model_name} "
                    f"model_seed={model_seed} split_seed={split_seed}"
                )
                print(
                    f"device={device} input_shape={input_shape} params={parameter_count} "
                    f"epochs={epochs} batch={batch_size} lr={learning_rate} wd={weight_decay}"
                )
                print(
                    f"split sizes: train={len(split_indices['train'])}, "
                    f"val={len(split_indices['val'])}, test={len(split_indices['test'])}"
                )
                history = train_model(
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    loaders[0],
                    loaders[1],
                    epochs,
                    device,
                    checkpoint_path,
                    PADDING_LENGTH,
                )
                if not checkpoint_path.exists():
                    raise RuntimeError(f"No best checkpoint produced: {checkpoint_path}")
                save_history_csv(history, history_path)
                checkpoint = torch_load_checkpoint(checkpoint_path, device)
                model.load_state_dict(checkpoint["model_state_dict"])
                y_pred, y_true, test_acc = _evaluate_model(model, loaders[2], device)
                with predictions_path.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        sample_indices=split_indices["test"],
                        y_true=np.asarray(y_true),
                        y_pred=np.asarray(y_pred),
                    )
                best_val_acc = float(checkpoint.get("best_val_acc", 0.0)) / 100.0
                print(f"best_val_acc={best_val_acc:.6f} test_acc={test_acc:.6f}")
        training_duration = time.time() - started
        artifacts = {
            "best_checkpoint.pth": {
                "size_bytes": checkpoint_path.stat().st_size,
            },
            "training_history.csv": {
                "size_bytes": history_path.stat().st_size,
            },
            "test_predictions.npz": {
                "size_bytes": predictions_path.stat().st_size,
            },
        }
        result = {
            "status": "ok",
            "dataset": dataset,
            "condition": condition,
            "model": model_name,
            "model_seed": model_seed,
            "split_seed": split_seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_val_acc": best_val_acc,
            "test_acc": float(test_acc),
            "train_size": len(split_indices["train"]),
            "val_size": len(split_indices["val"]),
            "test_size": len(split_indices["test"]),
            "input_shape": list(input_shape),
            "parameter_count": parameter_count,
            "preprocess_duration_sec": preprocess_duration,
            "training_duration_sec": training_duration,
            "total_duration_sec": preprocess_duration + training_duration,
            "config_fingerprint": config_fingerprint,
            "checkpoint": checkpoint_path.relative_to(condition_dir.parent).as_posix(),
            "history": history_path.relative_to(condition_dir.parent).as_posix(),
            "predictions": predictions_path.relative_to(condition_dir.parent).as_posix(),
            "artifacts": artifacts,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": "",
        }
        atomic_write_json(status_path, result)
        return result
    finally:
        del model, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def status_is_complete(
    condition_dir: Path, model_seed: int, config_fingerprint: str
) -> bool:
    run_dir = condition_dir / f"seed_{model_seed}"
    status = read_json(run_dir / "status.json")
    if not (
        status
        and status.get("status") == "ok"
        and status.get("config_fingerprint") == config_fingerprint
    ):
        return False
    artifact_metadata = status.get("artifacts")
    if not isinstance(artifact_metadata, dict):
        return False
    artifact_paths = {
        "best_checkpoint.pth": run_dir / "best_checkpoint.pth",
        "training_history.csv": run_dir / "training_history.csv",
        "test_predictions.npz": run_dir / "test_predictions.npz",
    }
    for name, path in artifact_paths.items():
        metadata = artifact_metadata.get(name)
        if (
            not isinstance(metadata, dict)
            or not path.is_file()
            or path.stat().st_size <= 0
            or path.stat().st_size != metadata.get("size_bytes")
        ):
            return False
    try:
        with np.load(artifact_paths["test_predictions.npz"], allow_pickle=False) as saved:
            if not {"sample_indices", "y_true", "y_pred"}.issubset(saved.files):
                return False
            if not (
                len(saved["sample_indices"])
                == len(saved["y_true"])
                == len(saved["y_pred"])
                == int(status.get("test_size", -1))
            ):
                return False
        with artifact_paths["training_history.csv"].open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            if sum(1 for _ in handle) < 2:
                return False
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    return True


def rebuild_summary(dataset_dir: Path, config_fingerprint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in ALL_CONDITIONS:
        condition_dir = dataset_dir / condition
        if not condition_dir.exists():
            continue
        for status_path in sorted(condition_dir.glob("seed_*/status.json")):
            status = read_json(status_path)
            if not status or status.get("config_fingerprint") != config_fingerprint:
                continue
            rows.append(status)
    rows.sort(key=lambda row: (ALL_CONDITIONS.index(row["condition"]), row["model_seed"]))
    successful = {
        row["condition"]: float(row["test_acc"])
        for row in rows
        if row.get("status") == "ok" and row.get("test_acc") is not None
    }
    linear = successful.get("linear_reference")
    robust = successful.get("robust_first50")
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        accuracy = output.get("test_acc")
        output["delta_vs_linear_pp"] = (
            "" if accuracy is None or linear is None else 100.0 * (float(accuracy) - linear)
        )
        output["delta_vs_robust_pp"] = (
            "" if accuracy is None or robust is None else 100.0 * (float(accuracy) - robust)
        )
        for key in ("input_shape",):
            if isinstance(output.get(key), list):
                output[key] = json.dumps(output[key])
        csv_rows.append(output)
    atomic_write_csv(dataset_dir / "summary.csv", csv_rows, SUMMARY_FIELDS)

    contrast_specs = {
        "robust_minus_linear": ("robust_first50", "linear_reference"),
        "robust_minus_common": ("robust_first50", "common_only"),
        "shared_minus_common": ("robust_shared_first50", "common_only"),
        "window_limited_minus_robust": (
            "robust_window_limited",
            "robust_first50",
        ),
        "fullspan50_minus_robust": ("robust_fullspan50", "robust_first50"),
        "common_minus_no_calibration": ("common_only", "no_calibration"),
        "detrend_minus_no_calibration": (
            "detrend_first50_only",
            "no_calibration",
        ),
    }
    contrasts = {}
    for name, (left, right) in contrast_specs.items():
        if left in successful and right in successful:
            contrasts[name] = {
                "left": left,
                "right": right,
                "delta_pp": 100.0 * (successful[left] - successful[right]),
            }
    atomic_write_json(
        dataset_dir / "contrasts.json",
        {"test_accuracy": successful, "contrasts": contrasts},
    )
    return rows


def remove_processed_cache(condition_dir: Path) -> None:
    # Exact files created by this script only; the reusable prefix cache remains.
    for name in (
        "processed.npy",
        "processed_done.npy",
        "processed_completion.json",
        "processed_progress.json",
        "diagnostics_partial.json",
    ):
        path = condition_dir / "cache" / name
        if path.exists():
            path.unlink()


def self_test() -> None:
    rng = np.random.default_rng(42)
    amplitude = rng.uniform(0.2, 2.0, size=(80, 30, 3))
    phase = np.cumsum(rng.normal(0.0, 0.05, size=(80, 30, 3)), axis=0)
    csi = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    mirror = robust_first50_mirror(csi)
    source = robust_phase_sanitization(csi)
    assert np.allclose(mirror, source, rtol=1e-7, atol=1e-8)
    short = csi[:2]
    assert np.allclose(
        robust_first50_mirror(short),
        robust_phase_sanitization(short),
        rtol=1e-7,
        atol=1e-8,
    )
    for condition in (
        "common_only",
        "detrend_first50_only",
        "detrend_fullspan50_only",
        "robust_shared_first50",
        "robust_window_limited",
        "robust_fullspan50",
    ):
        short_result = apply_phase_variant(short, condition, "gait")
        assert short_result.shape == short.shape
        assert np.all(np.isfinite(short_result))
    nearest = derive_nearest_indices("gait", 30, 15)
    source_nearest = execute_pipeline(
        csi, {"interpolate": INTERPOLATION_STEP}, dataset="gait"
    )
    assert np.array_equal(csi[:, nearest, :], source_nearest)
    for condition in ALL_CONDITIONS:
        result = apply_phase_variant(csi, condition, "gait")
        assert result.shape == csi.shape
        assert np.max(np.abs(np.abs(result) - np.abs(csi))) < 5e-6
        tail = explicit_tail(result, nearest, PADDING_LENGTH)
        assert tail.shape == (1500, 15, 6)
        assert tail.dtype == np.float32 and np.all(np.isfinite(tail))
    print("self-test: ok")


def resolve_conditions(args: argparse.Namespace) -> list[str]:
    if args.conditions:
        conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    else:
        conditions = list(CORE_CONDITIONS if args.suite == "core" else FULL_CONDITIONS)
    unknown = sorted(set(conditions) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; valid={ALL_CONDITIONS}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Duplicate conditions are not allowed")
    return conditions


def run_dataset(args: argparse.Namespace) -> None:
    dataset = args.dataset
    conditions = resolve_conditions(args)
    data_path = resolve_data_path(dataset, args.data_path)
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_dir = output_root / dataset
    entries, manifest_fingerprint, discovered_files = discover_entries(
        data_path, dataset, args.max_samples, args.split_seed
    )
    params = load_params(dataset)
    epochs = args.epochs or int(DATASET_DEFAULTS[dataset]["epochs"])
    batch_size = args.batch_size or int(params.get("batch", 32))
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(params.get("lr", 3e-4))
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(params.get("wd", 1e-3))
    )
    formal_training = (
        args.max_samples == 0 and not args.preprocess_only and not args.dry_run
    )
    device = resolve_device(
        args.device,
        allow_auto_cpu=(not formal_training or args.allow_cpu),
    )
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu"
    )
    wsdp_hash = hash_python_tree(WSDP_SRC / "wsdp")
    script_hash = sha256_file(Path(__file__).resolve())
    invariant_settings = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "data_path": str(data_path),
        "data_manifest_fingerprint": manifest_fingerprint,
        "selected_samples": len(entries),
        "max_samples": args.max_samples,
        "fixed_pipeline": {
            **PREFIX_STEPS,
            "phase_variant": "condition",
            "interpolate": INTERPOLATION_STEP,
            "normalize": {
                "method": "z-score",
                "return_phase_channels": True,
            },
            "padding_length": PADDING_LENGTH,
        },
        "model": args.model,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "diagnostic_samples": args.diagnostic_samples,
        "runtime": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "numpy_version": np.__version__,
        },
        "wsdp_src": str(WSDP_SRC),
        "wsdp_python_hash": wsdp_hash,
        "script_hash": script_hash,
        "representation": "float32 explicit [per-sample zscore amplitude, wrapped phase]",
    }
    config_fingerprint = fingerprint(invariant_settings)

    estimated_processed_bytes = (
        len(entries) * PADDING_LENGTH * 15 * 6 * np.dtype(np.float32).itemsize
    )
    print("=" * 80)
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"WSDP source: {WSDP_SRC}")
    print(f"Dataset: {dataset} at {data_path}")
    print(
        f"Files discovered={discovered_files}, parseable/selected={len(entries)}, "
        f"manifest={manifest_fingerprint[:12]}"
    )
    print(f"Conditions: {conditions}")
    print(
        f"seed(model/split)={args.model_seed}/{args.split_seed}, epochs={epochs}, "
        f"batch={batch_size}, workers={args.workers}, device={device} ({device_name})"
    )
    print(
        f"One processed condition is approximately "
        f"{gib(estimated_processed_bytes):.2f} GiB; "
        "conditions run sequentially."
    )
    print(f"Output: {dataset_dir}")
    print(f"Config fingerprint: {config_fingerprint}")
    if args.max_samples > 0:
        print("WARNING: --max-samples is a smoke-test subset, not a formal accuracy run.")
    if args.dry_run:
        print("dry-run: paths and configuration are valid; no output was written.")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    settings_path = dataset_dir / "settings.json"
    existing_settings = read_json(settings_path)
    if existing_settings is not None:
        if existing_settings.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError(
                "Existing output uses a different experiment configuration. "
                f"Refusing to mix results in {dataset_dir}. Choose a new --output-root."
            )
    else:
        atomic_write_json(
            settings_path,
            {
                "config_fingerprint": config_fingerprint,
                "invariant_settings": invariant_settings,
                "available_conditions": list(ALL_CONDITIONS),
                "core_conditions": list(CORE_CONDITIONS),
            },
        )
    append_jsonl(
        dataset_dir / "invocations.jsonl",
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": conditions,
            "preprocess_only": args.preprocess_only,
            "keep_processed": args.keep_processed,
            "workers": args.workers,
            "device": args.device,
            "config_fingerprint": config_fingerprint,
        },
    )

    pending = [
        condition
        for condition in conditions
        if not status_is_complete(
            dataset_dir / condition, args.model_seed, config_fingerprint
        )
    ]
    if not pending and not args.preprocess_only:
        print("Every requested condition is already complete; nothing to run.")
        rebuild_summary(dataset_dir, config_fingerprint)
        return

    prefix_id = make_prefix_id(manifest_fingerprint, wsdp_hash)
    check_disk_space(
        entries,
        dataset_dir,
        prefix_id,
        estimated_processed_bytes,
        pending if not args.preprocess_only else conditions,
        args.keep_processed,
        args.min_free_gib,
        args.skip_disk_check,
    )

    prefix_paths, manifest_rows, built_prefix_id = build_prefix_cache(
        entries,
        dataset,
        dataset_dir,
        manifest_fingerprint,
        wsdp_hash,
        args.workers,
    )
    if built_prefix_id != prefix_id:
        raise RuntimeError("Internal prefix-cache identity mismatch")
    labels, groups, unique_labels, unique_groups = encode_targets(entries)
    if len(unique_labels) < 2 and not args.preprocess_only:
        raise RuntimeError(
            f"Training requires >=2 classes, but selected data has {unique_labels}. "
            "Use --preprocess-only for a smoke test or run on the full server dataset."
        )
    dataset_metadata = {
        "dataset": dataset,
        "data_path": str(data_path),
        "discovered_files": discovered_files,
        "selected_samples": len(entries),
        "classes": unique_labels,
        "groups": unique_groups,
        "label_distribution": dict(Counter(entry["raw_label"] for entry in entries)),
        "group_distribution": dict(Counter(entry["raw_group"] for entry in entries)),
        "length": {
            "min": min(int(row["frames"]) for row in manifest_rows),
            "max": max(int(row["frames"]) for row in manifest_rows),
            "mean": float(np.mean([int(row["frames"]) for row in manifest_rows])),
        },
        "prefix_cache_id": prefix_id,
        "manifest_fingerprint": manifest_fingerprint,
    }
    atomic_write_json(dataset_dir / "dataset_metadata.json", dataset_metadata)
    atomic_write_csv(
        dataset_dir / "sample_manifest.csv", manifest_rows, manifest_rows[0].keys()
    )
    nearest_indices = run_equivalence_checks(prefix_paths, dataset, dataset_dir)
    split_indices = build_split_indices(
        labels,
        groups,
        dataset,
        args.split_seed,
        dataset_dir,
        config_fingerprint,
        require_class_coverage=formal_training,
    )
    failures: list[str] = []
    for condition in conditions:
        condition_dir = dataset_dir / condition
        if status_is_complete(condition_dir, args.model_seed, config_fingerprint):
            print(f"Skip completed condition: {condition}")
            continue
        condition_started = time.time()
        print("\n" + "=" * 80)
        print(f"Condition: {dataset} / {condition}")
        try:
            processed, preprocess_duration = build_processed_memmap(
                prefix_paths,
                condition,
                dataset,
                condition_dir,
                nearest_indices,
                args.workers,
                min(args.diagnostic_samples, len(prefix_paths)),
                config_fingerprint,
            )
            if args.preprocess_only:
                atomic_write_json(
                    condition_dir / "preprocess_status.json",
                    {
                        "status": "ok",
                        "condition": condition,
                        "shape": list(processed.shape),
                        "duration_sec": preprocess_duration,
                        "config_fingerprint": config_fingerprint,
                    },
                )
                del processed
                if not args.keep_processed:
                    remove_processed_cache(condition_dir)
                continue
            result = train_condition(
                processed,
                labels,
                unique_labels,
                split_indices,
                dataset,
                condition,
                condition_dir,
                args.model,
                args.model_seed,
                args.split_seed,
                epochs,
                batch_size,
                learning_rate,
                weight_decay,
                device,
                config_fingerprint,
                preprocess_duration,
            )
            print(
                f"Completed {condition}: val={result['best_val_acc']:.4f}, "
                f"test={result['test_acc']:.4f}"
            )
            del processed
            gc.collect()
            if not args.keep_processed:
                remove_processed_cache(condition_dir)
            rebuild_summary(dataset_dir, config_fingerprint)
        except BaseException as error:
            error_text = traceback.format_exc()
            run_dir = condition_dir / f"seed_{args.model_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
            atomic_write_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "dataset": dataset,
                    "condition": condition,
                    "model": args.model,
                    "model_seed": args.model_seed,
                    "split_seed": args.split_seed,
                    "config_fingerprint": config_fingerprint,
                    "total_duration_sec": time.time() - condition_started,
                    "error": repr(error),
                },
            )
            failures.append(condition)
            rebuild_summary(dataset_dir, config_fingerprint)
            print(error_text, file=sys.stderr)
            if not args.continue_on_error or isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = rebuild_summary(dataset_dir, config_fingerprint)
    completed = sorted(
        row["condition"]
        for row in rows
        if row.get("status") == "ok"
        and row.get("config_fingerprint") == config_fingerprint
    )
    atomic_write_json(
        dataset_dir / "completion.json",
        {
            "status": "failed" if failures else "ok",
            "requested_conditions": conditions,
            "completed_conditions": completed,
            "failed_conditions": failures,
            "config_fingerprint": config_fingerprint,
        },
    )
    if failures:
        raise RuntimeError(f"Conditions failed: {failures}")
    print(f"Finished. Summary: {dataset_dir / 'summary.csv'}")
    print(f"Contrasts: {dataset_dir / 'contrasts.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gait", "widar"), default="gait")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated condition list; overrides --suite",
    )
    parser.add_argument("--model", default="mlpmodel")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:1", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow --device auto to use CPU for a formal full-data training run",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means full data; positive values are only for smoke tests",
    )
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=5.0,
        help="free-space safety reserve used by the conservative disk preflight",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="warn instead of stopping when the conservative disk estimate fails",
    )
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--keep-processed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("workers", "diagnostic_samples"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.min_free_gib < 0:
        raise ValueError("--min-free-gib must be >= 0")
    for name in ("epochs", "batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("learning_rate", "weight_decay"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.list_conditions:
        print("core:", ",".join(CORE_CONDITIONS))
        print("full:", ",".join(FULL_CONDITIONS))
        return
    if args.self_test:
        self_test()
        return
    if args.dry_run:
        run_dataset(args)
        return
    output_root = Path(args.output_root).expanduser().resolve()
    lock_path, token = acquire_run_lock(output_root, args.dataset)
    try:
        run_dataset(args)
    finally:
        release_run_lock(lock_path, token)


if __name__ == "__main__":
    main()
