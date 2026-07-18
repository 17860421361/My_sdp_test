"""Validate why interpolation methods diverge after Robust calibration.

The four published conditions are reproduced with a fixed pipeline:

    wavelet -> IQR -> Robust -> INTERPOLATION -> z-score -> MLP

The same four methods are also run after linear phase calibration.  This paired
reference distinguishes "method X is generally weak" from a genuine
``Robust x interpolation`` interaction.  Additional causal controls separate
two mechanisms:

1. Cartesian complex cancellation: ``*_polar`` interpolates magnitude and
   unwrapped phase separately, then reconstructs the complex CSI.
2. Decimate-specific effects: ``direct_stride15`` selects 15 original tones
   without mixing.  Standard decimate applies an FIR low-pass on the reported
   tone-array axis and assumes uniform spacing, whereas the normal
   linear/cubic/nearest functions use the actual non-uniform IWL5300 tone grid.

Direct signal metrics and classification accuracy are both saved under
``result/ablations/robust_interpolation/<dataset>``.  Accuracy alone is not
treated as proof of cancellation.
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
from wsdp.algorithms.interpolation import decimate_antialias, interpolate_grid


CORE_CONDITIONS = (
    "nearest15_cartesian",
    "linear15_cartesian",
    "cubic15_cartesian",
    "decimate15_cartesian",
)
CONTROL_CONDITIONS = (
    "no_interpolation30",
    "linear15_polar",
    "cubic15_polar",
    "decimate15_polar",
    "direct_stride15",
)
ALL_CONDITIONS = CORE_CONDITIONS + CONTROL_CONDITIONS

SUMMARY_FIELDS = [
    "dataset",
    "condition",
    "calibration",
    "interpolation",
    "interpolation_family",
    "phase_aware",
    "uses_actual_tone_grid",
    "mixes_multiple_complex_tones",
    "normalization",
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
    "amplitude_energy_ratio_to_magnitude_path",
    "cancellation_ratio_p10",
    "cancellation_ratio_median",
    "fraction_ratio_lt_0.5",
    "reference_negative_fraction",
    "status",
    "error",
    "duration_sec",
]


def _target_count(csi: np.ndarray) -> int:
    return min(15, csi.shape[1])


def transform_interpolation(
    csi: np.ndarray,
    condition: str,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(complex output, magnitude-only reference)``.

    The reference applies the same spatial operator to ``abs(csi)``.  A small
    complex/reference magnitude ratio is direct evidence that Cartesian mixing
    cancelled vectors; it is not inferred from accuracy.
    """
    target = _target_count(csi)
    if condition == "no_interpolation30":
        return csi.copy(), np.abs(csi)
    if condition == "direct_stride15":
        if csi.shape[1] % target == 0:
            stride = csi.shape[1] // target
            indices = np.arange(0, csi.shape[1], stride, dtype=int)[:target]
        else:
            indices = np.rint(np.linspace(0, csi.shape[1] - 1, target)).astype(int)
        return csi[:, indices, :].copy(), np.abs(csi[:, indices, :])

    method = condition.split("15_", maxsplit=1)[0]
    is_polar = condition.endswith("_polar")
    if method in {"linear", "cubic", "nearest"}:
        magnitude = interpolate_grid(
            np.abs(csi), target_K=target, method=method, dataset=dataset
        )
        if is_polar:
            unwrapped_phase = np.unwrap(np.angle(csi), axis=1)
            phase = interpolate_grid(
                unwrapped_phase, target_K=target, method=method, dataset=dataset
            )
            safe_magnitude = np.maximum(magnitude, 0.0)
            return safe_magnitude * np.exp(1j * phase), magnitude
        output = interpolate_grid(csi, target_K=target, method=method, dataset=dataset)
        return output, magnitude

    if method == "decimate":
        magnitude = decimate_antialias(np.abs(csi), target_K=target, axis=1)
        if is_polar:
            unwrapped_phase = np.unwrap(np.angle(csi), axis=1)
            phase = decimate_antialias(unwrapped_phase, target_K=target, axis=1)
            safe_magnitude = np.maximum(magnitude, 0.0)
            return safe_magnitude * np.exp(1j * phase), magnitude
        return decimate_antialias(csi, target_K=target, axis=1), magnitude

    raise ValueError(f"Unknown interpolation condition: {condition}")


def interpolation_worker(csi: np.ndarray, condition: str, dataset: str) -> np.ndarray:
    return transform_interpolation(csi, condition, dataset)[0]


def normalize_worker(csi: np.ndarray) -> np.ndarray:
    return explicit_amplitude_phase(csi, "z-score")


def _adjacent_phase_metrics(samples: list[np.ndarray], limit: int) -> dict[str, float]:
    all_values = []
    early_values = []
    late_values = []
    for sample in samples[:limit]:
        phase = np.angle(sample)
        jumps = np.abs(np.angle(np.exp(1j * np.diff(phase, axis=1))))
        quarter = max(1, jumps.shape[0] // 4)
        all_values.append(jumps.reshape(-1))
        early_values.append(jumps[:quarter].reshape(-1))
        late_values.append(jumps[-quarter:].reshape(-1))
    values = np.concatenate(all_values)
    early = np.concatenate(early_values)
    late = np.concatenate(late_values)
    return {
        "samples": min(limit, len(samples)),
        "mean_rad": float(np.mean(values)),
        "p90_rad": float(np.percentile(values, 90)),
        "fraction_gt_pi_over_2": float(np.mean(values > np.pi / 2)),
        "early_quarter_mean_rad": float(np.mean(early)),
        "late_quarter_mean_rad": float(np.mean(late)),
        "late_to_early_ratio": float(np.mean(late) / max(np.mean(early), 1e-12)),
    }


def cancellation_metrics(
    samples: list[np.ndarray],
    condition: str,
    dataset: str,
    limit: int,
) -> dict[str, float]:
    ratios = []
    output_energy = 0.0
    reference_energy = 0.0
    reference_negative = 0
    reference_count = 0
    for sample in samples[:limit]:
        output, magnitude_reference = transform_interpolation(sample, condition, dataset)
        reference_negative += int(np.count_nonzero(magnitude_reference < 0))
        reference_count += magnitude_reference.size
        reference = np.abs(magnitude_reference)
        output_magnitude = np.abs(output)
        valid = reference > 1e-10
        ratios.append((output_magnitude[valid] / reference[valid]).reshape(-1))
        output_energy += float(np.sum(output_magnitude[valid]))
        reference_energy += float(np.sum(reference[valid]))
    ratio = np.concatenate(ratios)
    return {
        "diagnostic_samples": min(limit, len(samples)),
        "amplitude_energy_ratio_to_magnitude_path": float(
            output_energy / max(reference_energy, 1e-12)
        ),
        "cancellation_ratio_p10": float(np.percentile(ratio, 10)),
        "cancellation_ratio_median": float(np.median(ratio)),
        "fraction_ratio_lt_0.5": float(np.mean(ratio < 0.5)),
        "reference_negative_fraction": float(
            reference_negative / max(reference_count, 1)
        ),
    }


def condition_properties(condition: str) -> dict[str, object]:
    if condition == "no_interpolation30":
        family = "none"
    elif condition == "direct_stride15":
        family = "direct_selection"
    else:
        family = condition.split("15_", maxsplit=1)[0]
    return {
        "interpolation": condition,
        "interpolation_family": family,
        "phase_aware": condition.endswith("_polar"),
        "uses_actual_tone_grid": family in {"linear", "cubic", "nearest"},
        "mixes_multiple_complex_tones": condition.endswith("_cartesian")
        and family in {"linear", "cubic", "decimate"},
    }


def update_calibration_interactions(summary_path: Path, output_path: Path) -> None:
    """Pair Robust and linear-reference accuracy by method and model seed."""
    if not summary_path.exists():
        return
    records: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    with summary_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "ok":
                continue
            records[int(row["model_seed"])][
                (row["calibration"], row["interpolation"])
            ] = float(row["test_acc"])

    pairs = []
    for seed, values in sorted(records.items()):
        for interpolation in CORE_CONDITIONS:
            robust_key = ("robust", interpolation)
            linear_key = ("linear", interpolation)
            if robust_key not in values or linear_key not in values:
                continue
            pairs.append(
                {
                    "model_seed": seed,
                    "interpolation": interpolation,
                    "linear_test_acc": values[linear_key],
                    "robust_test_acc": values[robust_key],
                    "robust_penalty": values[robust_key] - values[linear_key],
                }
            )
    means = {}
    for interpolation in CORE_CONDITIONS:
        penalties = [
            item["robust_penalty"]
            for item in pairs
            if item["interpolation"] == interpolation
        ]
        if penalties:
            means[interpolation] = {
                "paired_seed_count": len(penalties),
                "mean_robust_penalty": float(np.mean(penalties)),
                "std_robust_penalty": float(np.std(penalties)),
            }
    write_json(
        output_path,
        {
            "definition": "robust_penalty = robust_test_acc - linear_test_acc",
            "paired_results": pairs,
            "per_interpolation_mean": means,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gait", "widar"), default="gait")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model", default="mlpmodel")
    parser.add_argument("--model-seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--robust-only",
        action="store_true",
        help="Skip the linear-calibration reference (not recommended for causal claims)",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(ALL_CONDITIONS),
        help="Comma-separated subset; defaults to four core methods and controls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}; valid={ALL_CONDITIONS}")
    data_path = resolve_data_path(args.dataset, args.data_path)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else REPO_ROOT / "result" / "ablations" / "robust_interpolation"
    )
    output_dir = output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    model_seeds = parse_int_list(args.model_seeds)
    padding_length = int(DATASET_DEFAULTS[args.dataset]["padding_length"])
    write_json(
        output_dir / "settings.json",
        {
            "question": (
                "Robust x interpolation interaction; Cartesian cancellation vs "
                "decimate-specific low-pass/grid effects"
            ),
            "dataset": args.dataset,
            "data_path": str(data_path),
            "fixed_prefix": fixed_prefix_steps(args.dataset),
            "calibrations": ["robust"] if args.robust_only else ["robust", "linear"],
            "fixed_normalization": "z-score after interpolation, explicit amplitude/phase",
            "robust_transforms": conditions,
            "linear_reference_transforms": (
                [] if args.robust_only else [item for item in conditions if item in CORE_CONDITIONS]
            ),
            "split_seed": args.split_seed,
            "model_seeds": model_seeds,
            "decimate_warning": (
                "standard decimate filters along array index and assumes uniform spacing; "
                "it does not use the actual non-uniform IWL5300 tone grid"
            ),
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
        stage_metrics = {
            "before_calibration": _adjacent_phase_metrics(
                prefix, min(args.diagnostic_samples, len(prefix))
            )
        }
        all_metrics = {}
        calibration_plan = [("robust", conditions)]
        if not args.robust_only:
            calibration_plan.append(
                ("linear", [item for item in conditions if item in CORE_CONDITIONS])
            )
        for calibration, transforms in calibration_plan:
            if not transforms:
                continue
            calibrated = parallel_execute_steps(
                executor,
                prefix,
                args.dataset,
                {"calibrate": {"method": calibration}},
            )
            stage_metrics[f"after_{calibration}"] = _adjacent_phase_metrics(
                calibrated, min(args.diagnostic_samples, len(calibrated))
            )
            write_json(output_dir / "calibration_adjacent_phase_metrics.json", stage_metrics)

            for transform in transforms:
                condition = f"{calibration}__{transform}"
                print("\n" + "=" * 80)
                print(f"Interpolation ablation: {args.dataset} / {condition}")
                start = time.time()
                metrics = cancellation_metrics(
                    calibrated,
                    transform,
                    args.dataset,
                    min(args.diagnostic_samples, len(calibrated)),
                )
                all_metrics[condition] = metrics
                write_json(output_dir / "interpolation_signal_metrics.json", all_metrics)

                worker = partial(
                    interpolation_worker, condition=transform, dataset=args.dataset
                )
                interpolated = list(executor.map(worker, calibrated, chunksize=1))
                final_data = list(executor.map(normalize_worker, interpolated, chunksize=1))
                del interpolated
                gc.collect()
                if args.skip_training:
                    del final_data
                    continue
                processed = resize_samples(final_data, padding_length)
                del final_data
                gc.collect()

                properties = condition_properties(transform)
                for model_seed in model_seeds:
                    key = (condition, str(model_seed))
                    if key in done:
                        print(f"Skip completed: {key}")
                        continue
                    row = {
                        "dataset": args.dataset,
                        "condition": condition,
                        "calibration": calibration,
                        **properties,
                        "normalization": "z-score_post_interpolation_explicit",
                        "model": args.model,
                        "model_seed": model_seed,
                        "split_seed": args.split_seed,
                        **metrics,
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
                        row.update(properties)
                        row.update(metrics)
                        row["dataset"] = args.dataset
                        row["calibration"] = calibration
                        row["normalization"] = "z-score_post_interpolation_explicit"
                        row["duration_sec"] = f"{time.time() - start:.2f}"
                    except Exception:
                        row["error"] = traceback.format_exc()
                        print(row["error"])
                    append_csv(summary_path, row, SUMMARY_FIELDS)
                    if row["status"] == "ok":
                        done.add(key)
                    update_calibration_interactions(
                        summary_path,
                        output_dir / "calibration_interactions.json",
                    )
                del processed
                gc.collect()

            del calibrated
            gc.collect()

        del prefix
        gc.collect()
    print(f"Finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
