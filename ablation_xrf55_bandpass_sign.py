"""XRF55 ablation for the signed Bandpass-output hypothesis.

Bandpass removes the DC baseline, so its real-valued output naturally contains
positive and negative deviations.  The current XRF55 path calls the generic
outlier function, which computes ``abs(csi)`` and returns that non-negative
magnitude for real input.  XRF55 train-split normalization also starts from
``abs(data)``.  This experiment separates those two sign-folding locations.

Cases
-----
``legacy_abs_iqr_absnorm``
    Current behavior: sign is lost in IQR; normalization uses magnitude.
``signed_iqr_absnorm``
    Preserve sign through IQR and complex-frequency resampling, but fold it at
    normalization.  This isolates the effect of folding before interpolation.
``signed_iqr_signednorm``
    Preserve sign through both IQR and train-split z-score.
``signed_no_iqr_signednorm``
    Preserve sign and remove IQR, testing whether IQR itself is destructive.

All cases use fs=200, 0.5-50 Hz, cubic15, the fixed XRF55 repetition split and
ResNet1D.  Results go to ``result/ablations/xrf55_bandpass_sign``.

Run from the repository root::

    python ablation_xrf55_bandpass_sign.py

The diagnostic CSV is generated before training and directly measures how
many values are negative and how much cross-subcarrier interpolation shrinks
signed signals relative to interpolation of their magnitudes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "SDP" / "test_xrf55"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import full_test_xrf55 as base  # noqa: E402

from wsdp.algorithms.amplitude import remove_outliers  # noqa: E402
from wsdp.algorithms.denoising_butterworth import butterworth_bandpass  # noqa: E402
from wsdp.algorithms.interpolation import interpolate_grid  # noqa: E402
from wsdp.processors.base_processor import (  # noqa: E402
    _parse_file_info_from_filename,
    _selector,
)
from wsdp.utils import resize_csi_to_fixed_length  # noqa: E402


np = base.np
MODEL_NAME = "resnet1d"
RESULT_DIR = ROOT / "result" / "ablations" / "xrf55_bandpass_sign"
SUMMARY_PATH = RESULT_DIR / "xrf55_bandpass_sign_summary.csv"
DIAGNOSTIC_PATH = RESULT_DIR / "xrf55_bandpass_sign_diagnostics.csv"

BRANCHES = [
    {
        "name": "legacy_abs_iqr_absnorm",
        "outlier_mode": "legacy_abs",
        "normalization_mode": "abs",
    },
    {
        "name": "signed_iqr_absnorm",
        "outlier_mode": "signed_iqr",
        "normalization_mode": "abs",
    },
    {
        "name": "signed_iqr_signednorm",
        "outlier_mode": "signed_iqr",
        "normalization_mode": "signed",
    },
    {
        "name": "signed_no_iqr_signednorm",
        "outlier_mode": "none",
        "normalization_mode": "signed",
    },
]

SUMMARY_FIELDS = [
    "case",
    "outlier_mode",
    "normalization_mode",
    "seed",
    "model",
    "status",
    "best_val_acc",
    "test_acc",
    "output_dir",
    "duration_sec",
    "error",
]


def resolve_data_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        ROOT / "sdp_dataset" / "xrf55" / "wifi",
        ROOT / "sdp_dataset" / "xrf55",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def stack_sample(csi_data) -> np.ndarray | None:
    frames = sorted(csi_data.frames, key=lambda frame: frame.timestamp)
    if not frames:
        return None
    csi = np.stack([frame.csi_array for frame in frames], axis=0)
    if csi.ndim == 2:
        csi = np.expand_dims(csi, -1)
    if csi.shape[0] < 2:
        return None
    if np.iscomplexobj(csi):
        if np.max(np.abs(np.imag(csi))) >= 1e-10:
            raise ValueError("XRF55 sign ablation expects real amplitude input")
        csi = np.real(csi)
    return csi


def signed_iqr(csi: np.ndarray) -> np.ndarray:
    """Clip magnitude with the existing IQR rule, then restore the sign."""
    clipped_magnitude = remove_outliers(csi, method="iqr", factor=1.5)
    return np.sign(csi) * clipped_magnitude


def transform_sample(csi: np.ndarray, outlier_mode: str) -> np.ndarray:
    filtered = butterworth_bandpass(
        csi,
        order=4,
        low_freq=0.5,
        high_freq=50.0,
        fs=200.0,
    )

    if outlier_mode == "legacy_abs":
        cleaned = remove_outliers(filtered, method="iqr", factor=1.5)
    elif outlier_mode == "signed_iqr":
        cleaned = signed_iqr(filtered)
    elif outlier_mode == "none":
        cleaned = filtered
    else:
        raise ValueError(f"unknown outlier mode: {outlier_mode}")

    return interpolate_grid(
        cleaned,
        target_K=15,
        method="cubic",
        dataset="xrf55",
    )


def _process_one(csi_data, outlier_mode: str):
    parsed = _parse_file_info_from_filename(csi_data.file_name, "xrf55")
    label, group = _selector(parsed, "xrf55")
    csi = stack_sample(csi_data)
    if csi is None:
        return None, None, None
    return transform_sample(csi, outlier_mode), label, group


def process_branch(raw_data, branch: dict, workers: int):
    worker = partial(_process_one, outlier_mode=branch["outlier_mode"])
    arrays, raw_labels, raw_groups = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for array, label, group in executor.map(worker, raw_data, chunksize=1):
            if array is not None:
                arrays.append(array)
                raw_labels.append(label)
                raw_groups.append(group)

    arrays = resize_csi_to_fixed_length(arrays, target_length=base.PADDING_LENGTH)
    labels_unique = sorted(set(raw_labels))
    groups_unique = sorted(set(raw_groups))
    label_map = {value: index for index, value in enumerate(labels_unique)}
    group_map = {value: index for index, value in enumerate(groups_unique)}
    labels = np.asarray([label_map[value] for value in raw_labels])
    groups = np.asarray([group_map[value] for value in raw_groups])
    return np.asarray(arrays), labels, groups, labels_unique


def normalize_from_training_split(split: tuple, mode: str) -> tuple:
    """Use exactly one scale fitted on repetitions 1-12, with optional sign."""
    train, val, test, train_y, val_y, test_y = split

    if mode == "abs":
        train_value, val_value, test_value = np.abs(train), np.abs(val), np.abs(test)
    elif mode == "signed":
        train_value, val_value, test_value = train, val, test
    else:
        raise ValueError(f"unknown normalization mode: {mode}")

    stat_axes = (0, 1) if train_value.ndim >= 2 else (0,)
    mean = np.mean(train_value, axis=stat_axes, keepdims=True)
    std = np.std(train_value, axis=stat_axes, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)

    return (
        (train_value - mean) / std,
        (val_value - mean) / std,
        (test_value - mean) / std,
        train_y,
        val_y,
        test_y,
    )


def split_and_normalize(data, labels, groups, mode: str) -> tuple:
    raw_split = base._create_data_split(
        data,
        labels,
        groups,
        test_split=base.TEST_SPLIT,
        val_split=base.VAL_SPLIT,
        seed=42,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset="xrf55",
        pipeline_steps={},
    )
    return normalize_from_training_split(raw_split, mode)


def append_summary(row: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def completed_case_seeds() -> set[tuple[str, int]]:
    if not SUMMARY_PATH.exists():
        return set()
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            (row["case"], int(row["seed"]))
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def run_training_case(branch: dict, seed: int, split: tuple, labels_unique, params):
    case_id = f"{branch['name']}_seed{seed}"
    output_dir = RESULT_DIR / f"{case_id}+{MODEL_NAME}"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    row = {
        "case": branch["name"],
        "outlier_mode": branch["outlier_mode"],
        "normalization_mode": branch["normalization_mode"],
        "seed": seed,
        "model": MODEL_NAME,
        "status": "failed",
        "best_val_acc": "",
        "test_acc": "",
        "output_dir": str(output_dir),
        "duration_sec": "",
        "error": "",
    }

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(base.Tee(sys.stdout, log)):
            try:
                base.set_seed(seed)
                pipeline_metadata = {"normalize": {"method": "z-score"}}
                batch_size = base.BATCH_SIZE or params.get("batch", 32)
                loaders = base.build_loaders(split, pipeline_metadata, batch_size)
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                model, device = base.create_registered_model(
                    MODEL_NAME,
                    len(labels_unique),
                    input_shape,
                )
                checkpoint = base.train_registered_model(
                    model,
                    device,
                    loaders,
                    params,
                    output_dir,
                    base.LEARNING_RATE,
                    base.WEIGHT_DECAY,
                    base.NUM_EPOCHS,
                    base.PADDING_LENGTH,
                    case_id,
                    MODEL_NAME,
                )
                val_acc, test_acc = base.evaluate_checkpoint(
                    model, device, loaders[2], checkpoint
                )
                row.update(
                    status="ok",
                    best_val_acc=val_acc,
                    test_acc=test_acc,
                )
            except Exception:
                traceback.print_exc()
                row["error"] = traceback.format_exc().splitlines()[-1]

    row["duration_sec"] = f"{time.time() - start:.2f}"
    return row


def diagnostic_row(csi_data) -> dict | None:
    csi = stack_sample(csi_data)
    if csi is None:
        return None
    filtered = butterworth_bandpass(
        csi, order=4, low_freq=0.5, high_freq=50.0, fs=200.0
    )
    clipped_abs = remove_outliers(filtered, method="iqr", factor=1.5)
    clipped_signed = np.sign(filtered) * clipped_abs
    legacy_interp = interpolate_grid(
        clipped_abs, target_K=15, method="cubic", dataset="xrf55"
    )
    signed_interp = interpolate_grid(
        clipped_signed, target_K=15, method="cubic", dataset="xrf55"
    )
    denominator = np.maximum(np.abs(legacy_interp), 1e-10)
    ratio = np.abs(signed_interp) / denominator
    return {
        "file_name": csi_data.file_name,
        "frames": csi.shape[0],
        "negative_fraction_after_bandpass": float(np.mean(filtered < 0)),
        "legacy_iqr_negative_fraction": float(np.mean(clipped_abs < 0)),
        "signed_iqr_negative_fraction": float(np.mean(clipped_signed < 0)),
        "fold_equivalence_max_error": float(
            np.max(np.abs(clipped_abs - np.abs(clipped_signed)))
        ),
        "interpolation_abs_ratio_mean": float(np.mean(ratio)),
        "interpolation_abs_ratio_median": float(np.median(ratio)),
        "interpolation_ratio_below_0_5": float(np.mean(ratio < 0.5)),
    }


def write_diagnostics(raw_data, sample_count: int) -> None:
    rows = []
    for csi_data in raw_data[:sample_count]:
        row = diagnostic_row(csi_data)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("no valid samples for diagnostics")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with DIAGNOSTIC_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSigned-output diagnostics:")
    for key in rows[0]:
        if key in {"file_name", "frames"}:
            continue
        values = [float(row[key]) for row in rows]
        print(f"  {key}: mean={statistics.mean(values):.6f}")
    print(f"diagnostic CSV: {DIAGNOSTIC_PATH}")


def print_aggregate() -> None:
    if not SUMMARY_PATH.exists():
        return
    grouped: dict[str, list[float]] = {}
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                grouped.setdefault(row["case"], []).append(float(row["test_acc"]))
    print("\nTraining aggregate:")
    for case, values in grouped.items():
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"  {case}: n={len(values)}, "
            f"test_acc={statistics.mean(values):.4f} +/- {std:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 49, 514])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--user-limit", type=int, default=3)
    parser.add_argument("--diagnostic-samples", type=int, default=30)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[branch["name"] for branch in BRANCHES],
        help="Run only selected cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"XRF55 data directory not found: {data_path}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base.DATA_PATH = data_path
    base.XRF55_USER_LIMIT = args.user_limit
    base.MODEL_NAME = MODEL_NAME
    base.NUM_EPOCHS = args.epochs
    base.PADDING_LENGTH = 1000
    base.RESULT_DIR = RESULT_DIR

    selected = [
        branch for branch in BRANCHES
        if not args.cases or branch["name"] in args.cases
    ]
    done = completed_case_seeds()
    pending = [
        (branch, seed)
        for branch in selected
        for seed in args.seeds
        if (branch["name"], seed) not in done
    ]

    print(f"data: {data_path}")
    print(f"cases: {[branch['name'] for branch in selected]}")
    print(f"seeds: {args.seeds}; epochs: {args.epochs}")
    print(f"pending training runs: {len(pending)}")

    if not pending and DIAGNOSTIC_PATH.exists():
        print_aggregate()
        return

    base.set_seed(42)
    raw_data = base.load_raw_data()
    write_diagnostics(raw_data, min(args.diagnostic_samples, len(raw_data)))
    if args.diagnostics_only:
        return

    params = base.load_params(base.DATASET_NAME)
    for branch in selected:
        branch_seeds = [
            seed for seed in args.seeds
            if (branch["name"], seed) not in done
        ]
        if not branch_seeds:
            continue

        print(f"\nPreparing branch: {json.dumps(branch, ensure_ascii=False)}")
        data, labels, groups, labels_unique = process_branch(
            raw_data, branch, args.workers
        )
        split = split_and_normalize(
            data, labels, groups, branch["normalization_mode"]
        )

        for seed in branch_seeds:
            try:
                row = run_training_case(branch, seed, split, labels_unique, params)
            finally:
                base.clear_cuda_cache()
            append_summary(row)

        del data, labels, groups, split
        gc.collect()

    print_aggregate()
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
