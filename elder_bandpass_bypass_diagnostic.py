"""Verify whether ElderAL segments actually enter the source Bandpass filter.

For the full-test parameters (order=4, 0.5--50 Hz, fs=1000 Hz), WSDP returns
the input unchanged whenever ``T < 3*max(len(a), len(b))+1``.  This script
derives that threshold from SciPy's filter coefficients and then checks every
raw ElderAL sample against the real WSDP output.

Outputs are written to ``result/ablations/elder_bandpass_bypass``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np
from scipy.signal import butter

from elder_ablation_common import (
    ABLATION_RESULT_ROOT,
    load_elder_records,
    record_to_array,
)


ORDER = 4
LOW_FREQ = 0.5
HIGH_FREQ = 50.0
FS = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 checks the full dataset; a positive value is useful for smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Import the exact source implementation only when the diagnostic runs.
    from wsdp.algorithms.denoising_butterworth import butterworth_bandpass

    records = load_elder_records()
    if args.max_samples > 0:
        records = records[: args.max_samples]

    nyquist = FS / 2.0
    b, a = butter(
        ORDER,
        [LOW_FREQ / nyquist, HIGH_FREQ / nyquist],
        btype="band",
    )
    source_min_length = 3 * max(len(a), len(b)) + 1

    output_dir = ABLATION_RESULT_ROOT / "elder_bandpass_bypass"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    length_histogram = Counter()

    for sample_index, record in enumerate(records, start=1):
        raw = record_to_array(record)
        filtered = butterworth_bandpass(
            raw,
            order=ORDER,
            low_freq=LOW_FREQ,
            high_freq=HIGH_FREQ,
            fs=FS,
        )
        length = int(raw.shape[0])
        predicted_bypass = length < source_min_length
        exact_same = bool(np.array_equal(filtered, raw, equal_nan=True))
        max_absolute_change = float(np.max(np.abs(filtered - raw)))
        length_histogram[length] += 1
        sample_rows.append(
            {
                "sample_index": sample_index,
                "file_name": str(record.file_name),
                "T": length,
                "source_min_length": source_min_length,
                "predicted_bypass": predicted_bypass,
                "output_exactly_equal_to_input": exact_same,
                "max_absolute_change": max_absolute_change,
            }
        )

        if sample_index % 100 == 0 or sample_index == len(records):
            print(f"Checked {sample_index}/{len(records)} ElderAL samples")

    bypass_rows = [row for row in sample_rows if row["predicted_bypass"]]
    eligible_rows = [row for row in sample_rows if not row["predicted_bypass"]]
    bypass_exact = sum(row["output_exactly_equal_to_input"] for row in bypass_rows)
    eligible_changed = sum(
        not row["output_exactly_equal_to_input"] for row in eligible_rows
    )

    summary = {
        "dataset": "elderAL",
        "parameters": {
            "order": ORDER,
            "low_freq": LOW_FREQ,
            "high_freq": HIGH_FREQ,
            "fs": FS,
        },
        "coefficient_lengths": {"b": len(b), "a": len(a)},
        "source_condition": f"return input unchanged when T < {source_min_length}",
        "samples": len(sample_rows),
        "predicted_bypass_samples": len(bypass_rows),
        "predicted_bypass_rate": len(bypass_rows) / max(len(sample_rows), 1),
        "bypassed_outputs_exactly_equal": bypass_exact,
        "bypassed_exact_match_rate": bypass_exact / max(len(bypass_rows), 1),
        "eligible_samples": len(eligible_rows),
        "eligible_outputs_changed": eligible_changed,
        "eligible_changed_rate": eligible_changed / max(len(eligible_rows), 1),
        "length_histogram": dict(sorted(length_histogram.items())),
    }

    samples_csv = output_dir / "bandpass_sample_diagnostics.csv"
    with samples_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    summary_json = output_dir / "bandpass_bypass_report.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nBandpass bypass verification")
    print(f"Derived source threshold: T < {source_min_length}")
    print(
        f"Predicted bypass: {len(bypass_rows)}/{len(sample_rows)} "
        f"({summary['predicted_bypass_rate']:.2%})"
    )
    print(
        f"Bypass outputs exactly equal to input: {bypass_exact}/{len(bypass_rows)} "
        f"({summary['bypassed_exact_match_rate']:.2%})"
    )
    print(
        f"Eligible outputs actually changed: {eligible_changed}/{len(eligible_rows)} "
        f"({summary['eligible_changed_rate']:.2%})"
    )
    print(f"Per-sample CSV: {samples_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
