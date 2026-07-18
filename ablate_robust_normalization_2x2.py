"""Clean 2x2 factorial test of Robust x amplitude normalization.

This experiment removes the ordering/representation confound in the current
320-pipeline benchmark.  Every condition uses exactly this order:

    fixed denoise -> fixed outlier removal -> {linear, Robust}
    -> nearest15 -> {z-score, min-max}
    -> explicit real [normalized amplitude, wrapped phase] -> MLP

The standard ``CSIDataset`` is intentionally not used for the final tensors:
for a real min-max amplitude/phase tensor it would otherwise call ``abs()`` and
fold negative phase.  The shared runner uses ``TensorDataset`` for all four
conditions, so only the two named factors change.

Results and the calculated interaction term are written under
``result/ablations/robust_normalization_2x2/<dataset>``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from robust_ablation_common import (
    DATASET_DEFAULTS,
    REPO_ROOT,
    append_csv,
    completed_keys,
    condition_input_stats,
    explicit_amplitude_phase,
    fixed_prefix_steps,
    load_raw_dataset,
    parallel_execute_steps,
    parse_int_list,
    resolve_data_path,
    resize_samples,
    train_one_seed,
    write_json,
)
from wsdp.algorithms import execute_pipeline


SUMMARY_FIELDS = [
    "dataset",
    "condition",
    "calibration",
    "normalization",
    "actual_order",
    "explicit_representation",
    "model",
    "model_seed",
    "split_seed",
    "best_val_acc",
    "test_acc",
    "input_shape",
    "train_size",
    "val_size",
    "test_size",
    "checkpoint",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_iqr",
    "phase_mean",
    "phase_std",
    "phase_to_amplitude_std_ratio",
    "status",
    "error",
    "duration_sec",
]


def calibrate_worker(csi: np.ndarray, calibration: str, dataset: str) -> np.ndarray:
    if calibration == "none":
        return csi.copy()
    return execute_pipeline(
        csi,
        {"calibrate": {"method": calibration}},
        dataset=dataset,
    )


def interpolate_worker(csi: np.ndarray, dataset: str) -> np.ndarray:
    return execute_pipeline(
        csi,
        {"interpolate": {"method": "nearest", "target_K": 15}},
        dataset=dataset,
    )


def normalize_worker(csi: np.ndarray, normalization: str) -> np.ndarray:
    return explicit_amplitude_phase(csi, normalization)


def update_factorial_effects(
    summary_path: Path,
    output_path: Path,
    control: str,
) -> None:
    """Calculate paired effects only from successful, same-model-seed runs."""
    if not summary_path.exists():
        return
    records: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    with summary_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok":
                continue
            key = (row["calibration"], row["normalization"])
            records[int(row["model_seed"])][key] = float(row["test_acc"])

    required = {
        (control, "z-score"),
        (control, "min-max"),
        ("robust", "z-score"),
        ("robust", "min-max"),
    }
    paired = []
    for seed, values in sorted(records.items()):
        if not required.issubset(values):
            continue
        robust_penalty_z = values[("robust", "z-score")] - values[(control, "z-score")]
        robust_penalty_mm = values[("robust", "min-max")] - values[(control, "min-max")]
        paired.append(
            {
                "model_seed": seed,
                "control_zscore_test_acc": values[(control, "z-score")],
                "control_minmax_test_acc": values[(control, "min-max")],
                "robust_zscore_test_acc": values[("robust", "z-score")],
                "robust_minmax_test_acc": values[("robust", "min-max")],
                "robust_effect_under_zscore": robust_penalty_z,
                "robust_effect_under_minmax": robust_penalty_mm,
                "interaction_extra_robust_loss_with_minmax": (
                    robust_penalty_mm - robust_penalty_z
                ),
            }
        )
    payload = {
        "definition": (
            "interaction = (robust-minmax - control-minmax) - "
            "(robust-zscore - control-zscore); a negative value means min-max "
            "makes the Robust penalty larger"
        ),
        "paired_seeds": paired,
    }
    if paired:
        payload["mean_robust_effect_under_zscore"] = float(
            np.mean([item["robust_effect_under_zscore"] for item in paired])
        )
        payload["mean_robust_effect_under_minmax"] = float(
            np.mean([item["robust_effect_under_minmax"] for item in paired])
        )
        payload["mean_interaction_extra_robust_loss_with_minmax"] = float(
            np.mean(
                [item["interaction_extra_robust_loss_with_minmax"] for item in paired]
            )
        )
    write_json(output_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gait", "widar"), default="widar")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--control", choices=("linear", "none"), default="linear")
    parser.add_argument("--model", default="mlpmodel")
    parser.add_argument("--model-seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrations = (args.control, "robust")
    normalizations = ("z-score", "min-max")
    data_path = resolve_data_path(args.dataset, args.data_path)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else REPO_ROOT / "result" / "ablations" / "robust_normalization_2x2"
    )
    output_dir = output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    model_seeds = parse_int_list(args.model_seeds)
    padding_length = int(DATASET_DEFAULTS[args.dataset]["padding_length"])
    actual_order = "denoise->outliers->calibrate->nearest15->normalize->explicit_amp_phase"
    write_json(
        output_dir / "settings.json",
        {
            "question": "Does min-max amplify the Robust penalty when order/representation are controlled?",
            "dataset": args.dataset,
            "data_path": str(data_path),
            "factors": {
                "calibration": list(calibrations),
                "normalization": list(normalizations),
            },
            "fixed_prefix": fixed_prefix_steps(args.dataset),
            "fixed_interpolation": {"method": "nearest", "target_K": 15},
            "actual_order_all_conditions": actual_order,
            "representation_all_conditions": "real [normalized amplitude, wrapped phase]",
            "dataset_wrapper": "TensorDataset (no implicit abs or representation conversion)",
            "split_seed": args.split_seed,
            "model_seeds": model_seeds,
        },
    )
    done = completed_keys(summary_path, ("condition", "model_seed"))

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=get_context("spawn"),
    ) as executor:
        raw, labels, groups, unique_labels, metadata = load_raw_dataset(
            args.dataset, data_path, executor, args.max_samples
        )
        write_json(output_dir / "dataset_metadata.json", metadata)
        prefix = parallel_execute_steps(
            executor, raw, args.dataset, fixed_prefix_steps(args.dataset)
        )
        del raw
        gc.collect()

        all_input_stats = {}
        for calibration in calibrations:
            calibration_worker = partial(
                calibrate_worker, calibration=calibration, dataset=args.dataset
            )
            calibrated = list(executor.map(calibration_worker, prefix, chunksize=1))
            interpolation_worker = partial(interpolate_worker, dataset=args.dataset)
            interpolated = list(executor.map(interpolation_worker, calibrated, chunksize=1))
            del calibrated
            gc.collect()

            for normalization in normalizations:
                condition = f"{calibration}__{normalization}"
                print("\n" + "=" * 80)
                print(f"Clean 2x2: {args.dataset} / {condition}")
                start = time.time()
                norm_worker = partial(normalize_worker, normalization=normalization)
                final_data = list(executor.map(norm_worker, interpolated, chunksize=1))
                stats = condition_input_stats(
                    final_data[: min(args.diagnostic_samples, len(final_data))]
                )
                all_input_stats[condition] = stats
                write_json(output_dir / "input_channel_stats.json", all_input_stats)
                if args.skip_training:
                    del final_data
                    continue
                processed = resize_samples(final_data, padding_length)
                del final_data
                gc.collect()

                for model_seed in model_seeds:
                    key = (condition, str(model_seed))
                    if key in done:
                        print(f"Skip completed: {key}")
                        continue
                    row = {
                        "dataset": args.dataset,
                        "condition": condition,
                        "calibration": calibration,
                        "normalization": normalization,
                        "actual_order": actual_order,
                        "explicit_representation": True,
                        "model": args.model,
                        "model_seed": model_seed,
                        "split_seed": args.split_seed,
                        **stats,
                        "status": "failed",
                        "error": "",
                        "duration_sec": "",
                    }
                    try:
                        trained = train_one_seed(
                            processed,
                            labels,
                            groups,
                            unique_labels,
                            dataset=args.dataset,
                            condition=condition,
                            model_seed=model_seed,
                            split_seed=args.split_seed,
                            output_dir=output_dir,
                            model_name=args.model,
                            epochs=args.epochs,
                        )
                        row.update(trained)
                        row.update(stats)
                        row["dataset"] = args.dataset
                        row["calibration"] = calibration
                        row["normalization"] = normalization
                        row["actual_order"] = actual_order
                        row["explicit_representation"] = True
                        row["duration_sec"] = f"{time.time() - start:.2f}"
                    except Exception:
                        row["error"] = traceback.format_exc()
                        print(row["error"])
                    append_csv(summary_path, row, SUMMARY_FIELDS)
                    if row["status"] == "ok":
                        done.add(key)
                    update_factorial_effects(
                        summary_path,
                        output_dir / "factorial_effects.json",
                        args.control,
                    )
                del processed
                gc.collect()
            del interpolated
            gc.collect()

        del prefix
        gc.collect()
    print(f"Finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
