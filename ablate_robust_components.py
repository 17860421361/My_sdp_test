"""Ablate the internal steps of Robust phase sanitization.

Question answered by this script
--------------------------------
Does accuracy fall because Robust removes the common phase, because it removes
one linear trend per subcarrier, or because the current implementation estimates
that trend from the first 50 frames and extrapolates it to the whole sequence?

Experiments are intentionally fixed to one preprocessing tail:

    fixed denoise -> fixed outlier removal -> PHASE VARIANT
    -> nearest15 -> z-score -> explicit [amplitude, phase] -> MLP

The group split is always seed 42.  Model seeds are varied independently.
Results are written under ``result/ablations/robust_components/<dataset>``.
No existing source file is modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
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
from wsdp.algorithms import execute_pipeline
from wsdp.algorithms.phase import robust_phase_sanitization


VARIANTS = (
    "linear_reference",
    "no_calibration",
    "common_only",
    "detrend_first50_only",
    "detrend_all_frames_only",
    "robust_first50",
    "robust_all_frames",
)

SUMMARY_FIELDS = [
    "dataset",
    "condition",
    "phase_variant",
    "slope_scope",
    "normalization",
    "interpolation",
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
    "status",
    "error",
    "duration_sec",
]


def _exact_theil_sen_slope(phases: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    """Exact median pairwise slope over the supplied frame indices."""
    if len(frame_indices) < 2:
        return np.zeros(phases.shape[1:], dtype=np.float64)
    left, right = np.triu_indices(len(frame_indices), k=1)
    i = frame_indices[left]
    j = frame_indices[right]
    denominators = (j - i).astype(np.float64)[:, None, None]
    slopes = (phases[j] - phases[i]) / denominators
    return np.median(slopes, axis=0)


def _all_frame_theil_sen_approx(
    phases: np.ndarray,
    anchor_count: int,
) -> np.ndarray:
    """Robust slope estimate in which every frame participates.

    Exact Theil-Sen is O(T^2) and is prohibitive for every sample/subcarrier.
    For T>200 this deterministic approximation pairs *every* frame with evenly
    spaced anchors spanning the entire action, then takes the median slope.  It
    therefore tests full-action coverage without changing the first-50 branch.
    The setting is saved with every run and can be increased from the CLI.
    """
    time_count = phases.shape[0]
    if time_count <= 200:
        return _exact_theil_sen_slope(phases, np.arange(time_count))
    anchors = np.unique(
        np.linspace(0, time_count - 1, min(anchor_count, time_count), dtype=int)
    )
    times = np.arange(time_count)
    slope_chunks = []
    for anchor in anchors:
        valid = times != anchor
        denominator = (times[valid] - anchor).astype(np.float64)[:, None, None]
        slope_chunks.append((phases[valid] - phases[anchor]) / denominator)
    return np.median(np.concatenate(slope_chunks, axis=0), axis=0)


def apply_phase_variant(
    csi: np.ndarray,
    variant: str,
    dataset: str,
    all_frame_anchors: int,
) -> np.ndarray:
    if variant == "linear_reference":
        return execute_pipeline(
            csi,
            {"calibrate": {"method": "linear"}},
            dataset=dataset,
        )
    if variant == "no_calibration":
        return csi.copy()
    if csi.ndim != 3 or np.isrealobj(csi):
        raise ValueError(f"Expected complex (T,F,A), got {csi.shape}, {csi.dtype}")

    phases = np.unwrap(np.angle(csi), axis=0)
    use_common = variant in {"common_only", "robust_first50", "robust_all_frames"}
    use_detrend = variant in {
        "detrend_first50_only",
        "detrend_all_frames_only",
        "robust_first50",
        "robust_all_frames",
    }
    use_all_frames = variant in {"detrend_all_frames_only", "robust_all_frames"}

    corrected = phases.copy()
    if use_common:
        corrected -= np.median(corrected, axis=1, keepdims=True)
    if use_detrend and corrected.shape[0] >= 3:
        if use_all_frames:
            slopes = _all_frame_theil_sen_approx(corrected, all_frame_anchors)
        else:
            frame_indices = np.arange(min(corrected.shape[0], 50))
            slopes = _exact_theil_sen_slope(corrected, frame_indices)
        times = np.arange(corrected.shape[0], dtype=np.float64)[:, None, None]
        corrected -= times * slopes[None, :, :]

    result = np.abs(csi) * np.exp(1j * corrected)
    return result.astype(csi.dtype, copy=False)


def phase_variant_worker(
    csi: np.ndarray,
    variant: str,
    dataset: str,
    all_frame_anchors: int,
) -> np.ndarray:
    return apply_phase_variant(csi, variant, dataset, all_frame_anchors)


def tail_worker(csi: np.ndarray, dataset: str) -> np.ndarray:
    interpolated = execute_pipeline(
        csi,
        {"interpolate": {"method": "nearest", "target_K": 15}},
        dataset=dataset,
    )
    return explicit_amplitude_phase(interpolated, "z-score")


def _circular_adjacent_jump(csi: np.ndarray) -> np.ndarray:
    phase = np.angle(csi)
    return np.abs(np.angle(np.exp(1j * np.diff(phase, axis=1))))


def variant_diagnostics(
    before: list[np.ndarray],
    after: list[np.ndarray],
    limit: int,
) -> dict[str, float]:
    amplitude_errors = []
    before_jumps = []
    after_jumps = []
    before_early_jumps = []
    before_late_jumps = []
    after_early_jumps = []
    after_late_jumps = []
    early_corrections = []
    late_corrections = []
    for source, result in zip(before[:limit], after[:limit]):
        amplitude_errors.append(float(np.max(np.abs(np.abs(result) - np.abs(source)))))
        source_jump = _circular_adjacent_jump(source)
        result_jump = _circular_adjacent_jump(result)
        quarter = max(1, source.shape[0] // 4)
        before_jumps.append(source_jump.reshape(-1))
        after_jumps.append(result_jump.reshape(-1))
        before_early_jumps.append(source_jump[:quarter].reshape(-1))
        before_late_jumps.append(source_jump[-quarter:].reshape(-1))
        after_early_jumps.append(result_jump[:quarter].reshape(-1))
        after_late_jumps.append(result_jump[-quarter:].reshape(-1))

        # Unwrap both temporal trajectories, remove their t=0 offset, and compare
        # the absolute correction near the start/end.  A large late/early ratio
        # is direct evidence that a slope estimated near the start is being
        # extrapolated into an increasingly large tail correction.
        source_phase = np.unwrap(np.angle(source), axis=0)
        result_phase = np.unwrap(np.angle(result), axis=0)
        correction = result_phase - source_phase
        correction -= correction[0:1]
        early_corrections.append(np.abs(correction[:quarter]).reshape(-1))
        late_corrections.append(np.abs(correction[-quarter:]).reshape(-1))
    before_values = np.concatenate(before_jumps)
    after_values = np.concatenate(after_jumps)
    before_early = np.concatenate(before_early_jumps)
    before_late = np.concatenate(before_late_jumps)
    after_early = np.concatenate(after_early_jumps)
    after_late = np.concatenate(after_late_jumps)
    early_correction = np.concatenate(early_corrections)
    late_correction = np.concatenate(late_corrections)
    return {
        "diagnostic_samples": min(limit, len(after)),
        "max_amplitude_preservation_error": float(max(amplitude_errors)),
        "adjacent_phase_jump_before_mean_rad": float(np.mean(before_values)),
        "adjacent_phase_jump_after_mean_rad": float(np.mean(after_values)),
        "adjacent_phase_jump_after_p90_rad": float(np.percentile(after_values, 90)),
        "adjacent_phase_jump_after_gt_pi_over_2": float(
            np.mean(after_values > (np.pi / 2))
        ),
        "adjacent_jump_before_late_to_early_ratio": float(
            np.mean(before_late) / max(np.mean(before_early), 1e-12)
        ),
        "adjacent_jump_after_late_to_early_ratio": float(
            np.mean(after_late) / max(np.mean(after_early), 1e-12)
        ),
        "phase_correction_early_quarter_mean_abs_rad": float(
            np.mean(early_correction)
        ),
        "phase_correction_late_quarter_mean_abs_rad": float(
            np.mean(late_correction)
        ),
        "phase_correction_late_to_early_ratio": float(
            np.mean(late_correction) / max(np.mean(early_correction), 1e-12)
        ),
    }


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
    parser.add_argument("--all-frame-anchors", type=int, default=12)
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated subset of phase variants",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; valid={VARIANTS}")
    if args.all_frame_anchors < 2:
        raise ValueError("--all-frame-anchors must be >= 2")

    data_path = resolve_data_path(args.dataset, args.data_path)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else REPO_ROOT / "result" / "ablations" / "robust_components"
    )
    output_dir = output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    model_seeds = parse_int_list(args.model_seeds)
    padding_length = int(DATASET_DEFAULTS[args.dataset]["padding_length"])

    settings = {
        "question": "common median vs per-subcarrier detrend; first50 vs all frames",
        "dataset": args.dataset,
        "data_path": str(data_path),
        "fixed_prefix": fixed_prefix_steps(args.dataset),
        "fixed_interpolation": {"method": "nearest", "target_K": 15},
        "fixed_normalization": "z-score after interpolation, explicit amplitude/phase",
        "variants": variants,
        "split_seed": args.split_seed,
        "model_seeds": model_seeds,
        "all_frame_estimator": (
            "exact all-pairs for T<=200; otherwise every frame paired with "
            f"{args.all_frame_anchors} evenly spaced full-action anchors"
        ),
    }
    write_json(output_dir / "settings.json", settings)
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

        # Prove that the custom first-50 implementation matches current source.
        ours = apply_phase_variant(
            prefix[0], "robust_first50", args.dataset, args.all_frame_anchors
        )
        source = robust_phase_sanitization(prefix[0])
        write_json(
            output_dir / "first50_source_equivalence.json",
            {
                "max_complex_absolute_error": float(np.max(np.abs(ours - source))),
                "max_amplitude_error": float(
                    np.max(np.abs(np.abs(ours) - np.abs(source)))
                ),
                "equivalent_at_rtol_1e-6_atol_1e-7": bool(
                    np.allclose(ours, source, rtol=1e-6, atol=1e-7)
                ),
            },
        )

        all_diagnostics = {}
        for variant in variants:
            print("\n" + "=" * 80)
            print(f"Robust component ablation: {args.dataset} / {variant}")
            start = time.time()
            worker = partial(
                phase_variant_worker,
                variant=variant,
                dataset=args.dataset,
                all_frame_anchors=args.all_frame_anchors,
            )
            calibrated = list(executor.map(worker, prefix, chunksize=1))
            diagnostics = variant_diagnostics(
                prefix, calibrated, min(args.diagnostic_samples, len(prefix))
            )
            all_diagnostics[variant] = diagnostics
            write_json(output_dir / "diagnostics.json", all_diagnostics)

            tail = partial(tail_worker, dataset=args.dataset)
            final_data = list(executor.map(tail, calibrated, chunksize=1))
            del calibrated
            gc.collect()
            if args.skip_training:
                del final_data
                continue
            processed = resize_samples(final_data, padding_length)
            del final_data
            gc.collect()

            for model_seed in model_seeds:
                condition = variant
                key = (condition, str(model_seed))
                if key in done:
                    print(f"Skip completed: {key}")
                    continue
                row = {
                    "dataset": args.dataset,
                    "condition": condition,
                    "phase_variant": variant,
                    "slope_scope": (
                        "all_frames"
                        if "all_frames" in variant
                        else "first50"
                        if "first50" in variant
                        else "none"
                    ),
                    "normalization": "z-score_post_interpolation_explicit",
                    "interpolation": "nearest15",
                    "model": args.model,
                    "model_seed": model_seed,
                    "split_seed": args.split_seed,
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
                    row["dataset"] = args.dataset
                    row["phase_variant"] = variant
                    row["slope_scope"] = (
                        "all_frames"
                        if "all_frames" in variant
                        else "first50"
                        if "first50" in variant
                        else "none"
                    )
                    row["normalization"] = "z-score_post_interpolation_explicit"
                    row["interpolation"] = "nearest15"
                    row["duration_sec"] = f"{time.time() - start:.2f}"
                except Exception:
                    row["error"] = traceback.format_exc()
                    print(row["error"])
                append_csv(summary_path, row, SUMMARY_FIELDS)
                if row["status"] == "ok":
                    done.add(key)
            del processed
            gc.collect()

        del prefix
        gc.collect()
    print(f"Finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
