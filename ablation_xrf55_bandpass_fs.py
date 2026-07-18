"""XRF55 Bandpass sampling-rate ablation.

Purpose
-------
Test only the sampling-rate hypothesis while keeping every other setting fixed:

    bandpass(0.5-50 Hz, order=4, fs in {1000, 200})
    -> IQR(1.5) -> train-split z-score -> cubic15 -> ResNet1D

The two cases use the same repetition split and model seeds.  If fs=200 is
consistently better across seeds, the old default fs=1000 was harmful.  If it
is still far below the Savgol baseline, sampling rate is only one cause rather
than the complete root cause.

Run on the server from the repository root::

    python ablation_xrf55_bandpass_fs.py

Results are written to ``result/ablations/xrf55_bandpass_fs``.  Existing
successful case/seed rows are skipped, so the script can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "SDP" / "test_xrf55"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import full_test_xrf55 as base  # noqa: E402


MODEL_NAME = "resnet1d"
RESULT_DIR = ROOT / "result" / "ablations" / "xrf55_bandpass_fs"
SUMMARY_PATH = RESULT_DIR / "xrf55_bandpass_fs_summary.csv"
SUMMARY_FIELDS = base.SUMMARY_FIELDS + ["sampling_rate_hz", "seed"]


def resolve_data_path(explicit: str | None) -> Path:
    """Support both the server's ``xrf55/wifi`` and the local flat layout."""
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


def build_cases(seeds: list[int]) -> list[dict]:
    cases: list[dict] = []
    index = 0
    for fs in (1000.0, 200.0):
        for seed in seeds:
            index += 1
            fs_label = int(fs)
            cases.append(
                {
                    "combo_index": index,
                    "combo_id": f"xrf55_bandpass_fs{fs_label}_seed{seed}",
                    "combo_name": (
                        f"bandpass_0.5-50_fs{fs_label}+iqr+z-score+cubic15"
                    ),
                    "denoise": f"bandpass_0.5-50_fs{fs_label}",
                    "outliers": "iqr",
                    "normalize": "z-score",
                    "interpolate": "cubic15",
                    "sampling_rate_hz": fs,
                    "seed": seed,
                    "pipeline_steps": {
                        "denoise": {
                            "method": "bandpass",
                            "order": 4,
                            "low_freq": 0.5,
                            "high_freq": 50.0,
                            "fs": fs,
                        },
                        "outliers": {"method": "iqr", "factor": 1.5},
                        "normalize": {"method": "z-score"},
                        "interpolate": {"method": "cubic", "target_K": 15},
                    },
                }
            )
    return cases


def load_successful_ids() -> set[str]:
    if not SUMMARY_PATH.exists():
        return set()
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            row["combo_id"]
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def append_row(row: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def print_aggregate() -> None:
    if not SUMMARY_PATH.exists():
        return
    groups: dict[float, list[float]] = {}
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            groups.setdefault(float(row["sampling_rate_hz"]), []).append(
                float(row["test_acc"])
            )

    print("\nSampling-rate ablation aggregate:")
    for fs in sorted(groups, reverse=True):
        values = groups[fs]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"  fs={fs:g} Hz | n={len(values)} | "
            f"test_acc={statistics.mean(values):.4f} +/- {std:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="XRF55 directory; auto-detected by default")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 49, 514])
    parser.add_argument(
        "--user-limit",
        type=int,
        default=3,
        help="Keep the same first-N-user scope as the existing experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"XRF55 data directory not found: {data_path}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base.RUN_NAME = "xrf55_bandpass_fs_ablation"
    base.MODEL_NAME = MODEL_NAME
    base.DATA_PATH = data_path
    base.XRF55_USER_LIMIT = args.user_limit
    base.RESULT_DIR = RESULT_DIR
    base.SUMMARY_PATH = SUMMARY_PATH
    base.NUM_EPOCHS = args.epochs

    cases = build_cases(args.seeds)
    done = load_successful_ids()
    pending = [case for case in cases if case["combo_id"] not in done]

    print(f"data: {data_path}")
    print(f"model: {MODEL_NAME}; epochs: {args.epochs}; seeds: {args.seeds}")
    print(f"pending: {len(pending)}/{len(cases)}")

    if pending:
        params = base.load_params(base.DATASET_NAME)
        base.set_seed(args.seeds[0])
        raw_data = base.load_raw_data()

        for case in pending:
            base.SEED = int(case["seed"])
            base.set_seed(base.SEED)
            try:
                row = base.run_one_combo(case, len(cases), raw_data, params)
            finally:
                base.clear_cuda_cache()
            row["sampling_rate_hz"] = case["sampling_rate_hz"]
            row["seed"] = case["seed"]
            append_row(row)

    print_aggregate()
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
