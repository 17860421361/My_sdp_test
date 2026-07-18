"""Measure what Hampel actually replaces in raw ElderAL action segments.

This is a signal-level diagnostic, not a training script.  For each requested
half-window it mirrors the WSDP Hampel implementation exactly and reports:

* value/frame/sample replacement rates;
* how often the local MAD is zero;
* how many replacements occur with a zero threshold;
* temporal total-variation and peak retention after filtering.

Outputs are written to ``result/ablations/elder_hampel_replacement``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np

from elder_ablation_common import (
    ABLATION_RESULT_ROOT,
    load_elder_records,
    record_to_array,
)


N_SIGMA = 3.0
CONSISTENCY_FACTOR = 1.4826


def _empty_stats(window_size: int) -> dict:
    return {
        "window_size": window_size,
        "full_window_frames": 2 * window_size + 1,
        "samples": 0,
        "frames": 0,
        "values": 0,
        "samples_with_replacement": 0,
        "frames_with_replacement": 0,
        "replaced_values": 0,
        "mad_zero_values": 0,
        "zero_mad_replacements": 0,
        "absolute_change_sum": 0.0,
        "raw_total_variation": 0.0,
        "filtered_total_variation": 0.0,
        "raw_peak_sum": 0.0,
        "filtered_peak_sum": 0.0,
    }


def _analyse_one(raw: np.ndarray, window_size: int) -> tuple[np.ndarray, dict]:
    """Vectorised equivalent of WSDP's channel-by-channel Hampel loop."""
    if np.iscomplexobj(raw):
        raise ValueError("ElderAL is expected to contain real amplitude CSI")

    source = np.asarray(raw)
    filtered = source.copy()
    changed = np.zeros(source.shape, dtype=bool)
    mad_zero = np.zeros(source.shape, dtype=bool)

    for time_index in range(source.shape[0]):
        lo = max(0, time_index - window_size)
        hi = min(source.shape[0], time_index + window_size + 1)
        local = source[lo:hi]
        median = np.median(local, axis=0)
        mad = np.median(np.abs(local - median), axis=0)
        threshold = N_SIGMA * CONSISTENCY_FACTOR * mad
        replace = np.abs(source[time_index] - median) > threshold

        filtered[time_index][replace] = median[replace]
        changed[time_index] = replace
        mad_zero[time_index] = mad == 0

    raw_tv = float(np.abs(np.diff(source, axis=0)).sum())
    filtered_tv = float(np.abs(np.diff(filtered, axis=0)).sum())
    frame_changed = np.any(changed.reshape(changed.shape[0], -1), axis=1)

    stats = {
        "samples": 1,
        "frames": int(source.shape[0]),
        "values": int(source.size),
        "samples_with_replacement": int(np.any(changed)),
        "frames_with_replacement": int(frame_changed.sum()),
        "replaced_values": int(changed.sum()),
        "mad_zero_values": int(mad_zero.sum()),
        "zero_mad_replacements": int(np.logical_and(changed, mad_zero).sum()),
        "absolute_change_sum": float(np.abs(filtered - source).sum()),
        "raw_total_variation": raw_tv,
        "filtered_total_variation": filtered_tv,
        "raw_peak_sum": float(np.max(source, axis=0).sum()),
        "filtered_peak_sum": float(np.max(filtered, axis=0).sum()),
    }
    return filtered, stats


def _merge(target: dict, update: dict) -> None:
    for key, value in update.items():
        target[key] += value


def _finalise(stats: dict) -> dict:
    values = max(stats["values"], 1)
    frames = max(stats["frames"], 1)
    samples = max(stats["samples"], 1)
    replaced = max(stats["replaced_values"], 1)
    raw_tv = stats["raw_total_variation"]
    raw_peaks = stats["raw_peak_sum"]

    row = stats.copy()
    row.update(
        {
            "replacement_rate": stats["replaced_values"] / values,
            "frame_replacement_rate": stats["frames_with_replacement"] / frames,
            "sample_replacement_rate": stats["samples_with_replacement"] / samples,
            "mad_zero_rate": stats["mad_zero_values"] / values,
            "zero_mad_share_of_replacements": (
                stats["zero_mad_replacements"] / replaced
                if stats["replaced_values"]
                else 0.0
            ),
            "mean_absolute_change": stats["absolute_change_sum"] / values,
            "total_variation_retention": (
                stats["filtered_total_variation"] / raw_tv if raw_tv else 1.0
            ),
            "peak_retention": (
                stats["filtered_peak_sum"] / raw_peaks if raw_peaks else 1.0
            ),
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
        help="Hampel half-window sizes (default: 1 2 3 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 analyses the full dataset; a positive value is useful for smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from wsdp.algorithms.amplitude import hampel_filter

    windows = sorted(set(args.windows))
    if not windows or min(windows) < 1:
        raise ValueError("all Hampel window sizes must be >= 1")

    records = load_elder_records()
    if args.max_samples > 0:
        records = records[: args.max_samples]

    output_dir = ABLATION_RESULT_ROOT / "elder_hampel_replacement"
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulators = {window: _empty_stats(window) for window in windows}
    length_histogram = Counter()
    source_equivalence_verified = False

    for sample_index, record in enumerate(records, start=1):
        raw = record_to_array(record)
        length_histogram[int(raw.shape[0])] += 1
        for window in windows:
            filtered, sample_stats = _analyse_one(raw, window)
            if sample_index == 1:
                source_filtered = hampel_filter(
                    raw,
                    window_size=window,
                    n_sigma=N_SIGMA,
                )
                if not np.array_equal(filtered, source_filtered, equal_nan=True):
                    raise AssertionError(
                        f"diagnostic does not match source Hampel for window={window}"
                    )
            _merge(accumulators[window], sample_stats)

        if sample_index == 1:
            source_equivalence_verified = True

        if sample_index % 100 == 0 or sample_index == len(records):
            print(f"Analysed {sample_index}/{len(records)} ElderAL samples")

    rows = [_finalise(accumulators[window]) for window in windows]
    csv_path = output_dir / "hampel_replacement_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset": "elderAL",
        "n_sigma": N_SIGMA,
        "analysed_samples": len(records),
        "source_equivalence_verified_on_first_sample": source_equivalence_verified,
        "length_histogram": dict(sorted(length_histogram.items())),
        "metrics": rows,
        "interpretation": {
            "replacement_rate": "fraction of scalar CSI amplitudes replaced",
            "mad_zero_rate": "fraction whose local Hampel threshold is exactly zero",
            "zero_mad_share_of_replacements": "replacements made under a zero threshold",
            "total_variation_retention": "<1 means temporal changes were removed",
            "peak_retention": "<1 means local/global amplitude peaks were reduced",
        },
    }
    json_path = output_dir / "hampel_replacement_report.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\nwindow | full | replaced | MAD=0 | zero-MAD among replaced | TV retained")
    for row in rows:
        print(
            f"{row['window_size']:>6} | {row['full_window_frames']:>4} | "
            f"{row['replacement_rate']:.2%} | {row['mad_zero_rate']:.2%} | "
            f"{row['zero_mad_share_of_replacements']:.2%} | "
            f"{row['total_variation_retention']:.2%}"
        )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
