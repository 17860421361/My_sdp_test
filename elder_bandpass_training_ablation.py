"""Train the ElderAL Bandpass-versus-no-denoise accuracy control.

The downstream pipeline is fixed to ``IQR + min-max + linear64`` and the model
is CSI-Time.  Combine this summary with the per-sample bypass report from
``elder_bandpass_bypass_diagnostic.py``: if most samples bypass and the two
accuracies are close, the apparently good Bandpass score cannot be attributed
to Bandpass filtering those short samples.

Results are written to ``result/ablations/elder_bandpass_training``.
"""

from __future__ import annotations

import argparse

from elder_ablation_common import make_combo, run_csitime_ablation


def build_combinations() -> list[dict]:
    denoisers = [
        ("none", None),
        (
            "bandpass_o4_0.5-50_fs1000",
            {
                "method": "bandpass",
                "order": 4,
                "low_freq": 0.5,
                "high_freq": 50.0,
                "fs": 1000.0,
            },
        ),
    ]
    return [
        make_combo(
            combo_index=index,
            combo_id=f"elder_bandpass_ablation_{index:02d}",
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
        "elder_bandpass_training",
        build_combinations(),
        epochs=args.epochs,
        model_seeds=args.seeds,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
