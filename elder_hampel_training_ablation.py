"""Train the causal ElderAL Hampel-window ablation with CSI-Time.

Everything after denoising is fixed to ``IQR + min-max + linear64``.  The
experiment includes true no-denoise and Savgol controls, four Hampel window
sizes, and two wider Hampel thresholds.  The latter distinguish ordinary
over-replacement from the special MAD=0 case (multiplying a zero threshold by
six still leaves it at zero).

Results are written to ``result/ablations/elder_hampel_training``.
"""

from __future__ import annotations

import argparse

from elder_ablation_common import make_combo, run_csitime_ablation


def build_combinations() -> list[dict]:
    denoisers = [
        ("none", None),
        ("savgol_w7_p3", {"method": "savgol", "window_length": 7, "polyorder": 3}),
        ("hampel_w1_s3", {"method": "hampel", "window_size": 1, "n_sigma": 3.0}),
        ("hampel_w2_s3", {"method": "hampel", "window_size": 2, "n_sigma": 3.0}),
        ("hampel_w3_s3", {"method": "hampel", "window_size": 3, "n_sigma": 3.0}),
        ("hampel_w5_s3", {"method": "hampel", "window_size": 5, "n_sigma": 3.0}),
        ("hampel_w2_s6", {"method": "hampel", "window_size": 2, "n_sigma": 6.0}),
        ("hampel_w5_s6", {"method": "hampel", "window_size": 5, "n_sigma": 6.0}),
    ]
    return [
        make_combo(
            combo_index=index,
            combo_id=f"elder_hampel_ablation_{index:02d}",
            denoise_name=name,
            denoise_config=config,
        )
        for index, (name, config) in enumerate(denoisers, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 49, 514])
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_csitime_ablation(
        "elder_hampel_training",
        build_combinations(),
        epochs=args.epochs,
        model_seeds=args.seeds,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
