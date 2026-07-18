"""Shared runtime for the Robust phase-calibration ablation experiments.

This module deliberately contains only data loading, deterministic splitting and
training utilities.  The three experiment definitions live in separate scripts
at the repository root so that each causal question can be run and inspected on
its own.

The final model representation is always an explicit real-valued tensor.  This
is important for fair ablations: it prevents :class:`CSIDataset` from taking an
extra ``abs()`` depending on the declared normalization method.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import sys
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsdp_mplconfig")

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent


def _find_wsdp_src() -> Path:
    candidates = (
        REPO_ROOT
        / "SDP"
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        REPO_ROOT
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
    )
    for candidate in candidates:
        if (candidate / "wsdp").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find the local WSDP source tree. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


WSDP_SRC = _find_wsdp_src()
sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp import readers  # noqa: E402
from wsdp.algorithms import execute_pipeline  # noqa: E402
from wsdp.algorithms.amplitude import normalize_amplitude  # noqa: E402
from wsdp.core import _create_data_split, _evaluate_model  # noqa: E402
from wsdp.models import create_model  # noqa: E402
from wsdp.processors.base_processor import (  # noqa: E402
    _parse_file_info_from_filename,
    _selector,
)
from wsdp.utils import load_params, resize_csi_to_fixed_length, train_model  # noqa: E402


DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "gait": {
        "relative_path": Path("Gait_Dataset") / "CSI_Gait",
        "denoise": {"method": "wavelet"},
        "outliers": {"method": "iqr", "factor": 1.5},
        "epochs": 60,
        "padding_length": 1500,
    },
    "widar": {
        "relative_path": Path("widar_common3"),
        "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
        "outliers": {"method": "z-score", "factor": 3.0},
        "epochs": 80,
        "padding_length": 1500,
    },
}


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one seed is required")
    return result


def resolve_data_path(dataset: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    relative = DATASET_DEFAULTS[dataset]["relative_path"]
    candidates = (
        REPO_ROOT / "sdp_dataset" / relative,
        REPO_ROOT.parent / "sdp_dataset" / relative,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the normal server-layout path so the error is immediately useful.
    return candidates[0].resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_raw_sample(csi_data, dataset: str):
    """Convert one reader object to ``(T,F,A)`` and parse label/group."""
    parsed = _parse_file_info_from_filename(csi_data.file_name, dataset)
    if parsed is None:
        return None, None, None
    label, group = _selector(parsed, dataset)
    frames = sorted(csi_data.frames, key=lambda frame: frame.timestamp)
    tensors = [frame.csi_array for frame in frames]
    if not tensors:
        return None, None, None
    csi = np.stack(tensors, axis=0)
    if csi.ndim == 2:
        csi = np.expand_dims(csi, -1)
    if csi.ndim != 3 or csi.shape[0] < 2:
        return None, None, None
    return csi, label, group


def load_raw_dataset(
    dataset: str,
    data_path: Path,
    executor: ProcessPoolExecutor,
    max_samples: int | None = None,
):
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    reader_items = readers.load_data(str(data_path), dataset)
    if max_samples is not None:
        reader_items = reader_items[:max_samples]
    worker = partial(parse_raw_sample, dataset=dataset)
    arrays: list[np.ndarray] = []
    raw_labels: list[Any] = []
    raw_groups: list[Any] = []
    for csi, label, group in executor.map(worker, reader_items, chunksize=1):
        if csi is not None:
            arrays.append(csi)
            raw_labels.append(label)
            raw_groups.append(group)
    if not arrays:
        raise RuntimeError("No valid CSI samples were parsed")

    unique_labels = sorted(set(raw_labels))
    unique_groups = sorted(set(raw_groups))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    group_map = {group: index for index, group in enumerate(unique_groups)}
    labels = np.asarray([label_map[label] for label in raw_labels], dtype=np.int64)
    groups = np.asarray([group_map[group] for group in raw_groups], dtype=np.int64)
    metadata = {
        "dataset": dataset,
        "data_path": str(data_path),
        "sample_count": len(arrays),
        "raw_sample_shape": list(arrays[0].shape),
        "unique_labels": [str(item) for item in unique_labels],
        "label_distribution": {str(k): int(v) for k, v in Counter(raw_labels).items()},
        "group_distribution": {str(k): int(v) for k, v in Counter(raw_groups).items()},
    }
    print(
        f"Loaded {len(arrays)} valid {dataset} samples; "
        f"classes={len(unique_labels)}, groups={len(unique_groups)}, "
        f"first_shape={arrays[0].shape}"
    )
    return arrays, labels, groups, unique_labels, metadata


def execute_steps_worker(csi: np.ndarray, dataset: str, steps: dict) -> np.ndarray:
    return execute_pipeline(csi, steps, dataset=dataset)


def parallel_execute_steps(
    executor: ProcessPoolExecutor,
    data: Iterable[np.ndarray],
    dataset: str,
    steps: dict,
) -> list[np.ndarray]:
    worker = partial(execute_steps_worker, dataset=dataset, steps=steps)
    return list(executor.map(worker, data, chunksize=1))


def fixed_prefix_steps(dataset: str) -> dict:
    defaults = DATASET_DEFAULTS[dataset]
    return {
        "denoise": dict(defaults["denoise"]),
        "outliers": dict(defaults["outliers"]),
    }


def explicit_amplitude_phase(csi: np.ndarray, method: str) -> np.ndarray:
    """Return explicit real ``[normalized amplitude, wrapped phase]`` channels."""
    return normalize_amplitude(csi, method=method, return_phase_channels=True)


def resize_samples(data: list[np.ndarray], padding_length: int) -> np.ndarray:
    resized = resize_csi_to_fixed_length(data, target_length=padding_length)
    result = np.asarray(resized)
    if np.iscomplexobj(result):
        raise RuntimeError(
            "The training representation must be explicit real amplitude/phase channels"
        )
    return result.astype(np.float32, copy=False)


def split_processed_data(
    processed: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    dataset: str,
    split_seed: int,
):
    return _create_data_split(
        processed,
        labels,
        groups,
        test_split=0.3,
        val_split=0.5,
        seed=split_seed,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=dataset,
        pipeline_steps={},
    )


def build_tensor_loaders(split, batch_size: int, model_seed: int):
    train_data, val_data, test_data, train_y, val_y, test_y = split

    def make_loader(data, labels, shuffle: bool):
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(data)).float(),
            torch.from_numpy(np.asarray(labels)).long(),
        )
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(model_seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    return (
        make_loader(train_data, train_y, True),
        make_loader(val_data, val_y, False),
        make_loader(test_data, test_y, False),
    )


def _save_history(history: dict, path: Path) -> None:
    keys = list(history.keys())
    length = max((len(history[key]) for key in keys), default=0)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for index in range(length):
            writer.writerow(
                {
                    key: history[key][index] if index < len(history[key]) else ""
                    for key in keys
                }
            )


def train_one_seed(
    processed: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    unique_labels: list,
    *,
    dataset: str,
    condition: str,
    model_seed: int,
    split_seed: int,
    output_dir: Path,
    model_name: str = "mlpmodel",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> dict[str, Any]:
    """Train one model seed while keeping the group split seed independent."""
    params = load_params(dataset)
    epochs = epochs if epochs is not None else int(params.get("num_epochs", 60))
    batch_size = batch_size if batch_size is not None else int(params.get("batch", 32))
    learning_rate = (
        learning_rate if learning_rate is not None else float(params.get("lr", 3e-4))
    )
    weight_decay = (
        weight_decay if weight_decay is not None else float(params.get("wd", 1e-3))
    )

    split = split_processed_data(processed, labels, groups, dataset, split_seed)
    set_seed(model_seed)
    loaders = build_tensor_loaders(split, batch_size, model_seed)
    input_shape = tuple(loaders[0].dataset.tensors[0].shape[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name,
        num_classes=len(unique_labels),
        input_shape=input_shape,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    run_dir = output_dir / condition / f"seed_{model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_checkpoint.pth"
    print(
        f"Training {condition}, model_seed={model_seed}, split_seed={split_seed}, "
        f"input_shape={input_shape}, device={device}, epochs={epochs}"
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
        checkpoint,
        int(params.get("padding_length", 1500)),
    )
    _save_history(history, run_dir / "training_history.csv")
    if not checkpoint.exists():
        raise RuntimeError(f"No best checkpoint was produced: {checkpoint}")
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    _, _, test_acc = _evaluate_model(model, loaders[2], device)
    best_val_acc = float(saved.get("best_val_acc", 0.0)) / 100.0
    result = {
        "condition": condition,
        "model": model_name,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "best_val_acc": best_val_acc,
        "test_acc": float(test_acc),
        "input_shape": json.dumps(input_shape),
        "train_size": len(loaders[0].dataset),
        "val_size": len(loaders[1].dataset),
        "test_size": len(loaders[2].dataset),
        "checkpoint": str(checkpoint),
        "status": "ok",
        "error": "",
    }
    del model, loaders, split
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def completed_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return {
            tuple(row.get(field, "") for field in key_fields)
            for row in csv.DictReader(file)
            if row.get("status") == "ok"
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def condition_input_stats(data: list[np.ndarray]) -> dict[str, float]:
    """Summarize explicit amplitude/phase channel scale before padding."""
    amp_values = []
    phase_values = []
    for sample in data:
        channels = sample.shape[-1] // 2
        amp_values.append(sample[..., :channels].reshape(-1))
        phase_values.append(sample[..., channels:].reshape(-1))
    amplitude = np.concatenate(amp_values)
    phase = np.concatenate(phase_values)
    return {
        "amplitude_mean": float(np.mean(amplitude)),
        "amplitude_std": float(np.std(amplitude)),
        "amplitude_iqr": float(np.percentile(amplitude, 75) - np.percentile(amplitude, 25)),
        "phase_mean": float(np.mean(phase)),
        "phase_std": float(np.std(phase)),
        "phase_to_amplitude_std_ratio": float(
            np.std(phase) / max(np.std(amplitude), 1e-12)
        ),
    }
