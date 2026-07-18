"""Shared helpers for the standalone ElderAL ablation scripts.

The actual ablations live in separate scripts at the repository root.  This
module only centralises path handling, raw-sample extraction, and the existing
CSI-Time training entry point so that every ablation uses exactly the same
split, model, padding length, and evaluation code as ``full_test_elder.py``.
"""

from __future__ import annotations

import contextlib
import csv
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
ELDER_TEST_DIR = REPO_ROOT / "SDP" / "test_elderAL"
WSDP_SRC = (
    REPO_ROOT
    / "SDP"
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src"
)
DEFAULT_DATA_PATH = REPO_ROOT / "sdp_dataset" / "elderAL"
ABLATION_RESULT_ROOT = REPO_ROOT / "result" / "ablations"

for import_path in (ELDER_TEST_DIR, WSDP_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def elder_data_path() -> Path:
    """Return the server dataset path, with an optional environment override."""
    return Path(os.environ.get("ELDERAL_DATA_PATH", DEFAULT_DATA_PATH)).expanduser()


def load_elder_records(data_path: Optional[Path] = None):
    """Load ElderAL ``CSIData`` records with the same reader as training."""
    # Keep this import lazy: signal-only helpers can be imported on a local
    # machine that does not have the server's PyTorch environment installed.
    from wsdp import readers

    path = Path(data_path) if data_path is not None else elder_data_path()
    if not path.is_dir():
        raise FileNotFoundError(
            f"ElderAL data directory not found: {path}\n"
            "Expected sdp_dataset/elderAL at the repository root.  "
            "Alternatively set ELDERAL_DATA_PATH."
        )
    records = readers.load_data(str(path), "elderAL")
    if not records:
        raise RuntimeError(f"No ElderAL samples were loaded from {path}")
    return records


def record_to_array(record) -> np.ndarray:
    """Convert one ``CSIData`` record to the processor's (T, F, A) array."""
    frames = sorted(record.frames, key=lambda frame: frame.timestamp)
    if not frames:
        raise ValueError(f"record has no frames: {record.file_name}")

    csi = np.stack([frame.csi_array for frame in frames], axis=0)
    if csi.ndim == 2:
        csi = np.expand_dims(csi, -1)
    if csi.ndim != 3:
        raise ValueError(
            f"expected (T, F, A), got {csi.shape}: {record.file_name}"
        )
    return csi


def make_combo(
    combo_index: int,
    combo_id: str,
    denoise_name: str,
    denoise_config: Optional[dict],
    *,
    outlier_name: str = "iqr",
    outlier_config: Optional[dict] = None,
    normalize_name: str = "min-max",
    normalize_config: Optional[dict] = None,
    interpolate_name: str = "linear64",
    interpolate_config: Optional[dict] = None,
) -> dict:
    """Build one training row while allowing a true no-denoise condition."""
    if outlier_config is None:
        outlier_config = {"method": "iqr", "factor": 1.5}
    if normalize_config is None:
        normalize_config = {"method": "min-max"}
    if interpolate_config is None:
        interpolate_config = {"method": "linear", "target_K": 64}

    pipeline_steps = {
        "outliers": outlier_config.copy(),
        "normalize": normalize_config.copy(),
        "interpolate": interpolate_config.copy(),
    }
    if denoise_config is not None:
        pipeline_steps = {"denoise": denoise_config.copy(), **pipeline_steps}

    combo_name = "+".join(
        [denoise_name, outlier_name, normalize_name, interpolate_name]
    )
    return {
        "combo_index": combo_index,
        "combo_id": combo_id,
        "combo_name": combo_name,
        "denoise": denoise_name,
        "outliers": outlier_name,
        "normalize": normalize_name,
        "interpolate": interpolate_name,
        "pipeline_steps": pipeline_steps,
    }


def _successful_combo_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            row["combo_id"]
            for row in csv.DictReader(handle)
            if row.get("combo_id")
            and row.get("model") == "csitime"
            and row.get("status") == "ok"
        }


def _append_summary(summary_path: Path, fieldnames: list[str], row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_seed_aggregate(
    summary_path: Path,
    aggregate_path: Path,
    templates: list[dict],
    model_seeds: list[int],
) -> None:
    """Write mean/std across model seeds for report-ready comparison."""
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        # Keep the last successful row if a previously failed seed was rerun.
        successful_rows = {
            row["combo_id"]: row
            for row in csv.DictReader(handle)
            if row.get("combo_id") and row.get("status") == "ok"
        }

    aggregate_rows = []
    for template in templates:
        rows = []
        completed_seeds = []
        for seed in model_seeds:
            combo_id = f"{template['combo_id']}_seed{seed}"
            if combo_id in successful_rows:
                rows.append(successful_rows[combo_id])
                completed_seeds.append(seed)

        if not rows:
            continue
        val_scores = np.asarray(
            [float(row["best_val_acc"]) for row in rows], dtype=float
        )
        test_scores = np.asarray(
            [float(row["test_acc"]) for row in rows], dtype=float
        )
        aggregate_rows.append(
            {
                "base_combo_id": template["combo_id"],
                "base_combo_name": template["combo_name"],
                "denoise": template["denoise"],
                "outliers": template["outliers"],
                "normalize": template["normalize"],
                "interpolate": template["interpolate"],
                "n_seeds": len(rows),
                "model_seeds": ",".join(map(str, completed_seeds)),
                "mean_best_val_acc": float(val_scores.mean()),
                "std_best_val_acc": float(val_scores.std(ddof=1))
                if len(val_scores) > 1
                else 0.0,
                "mean_test_acc": float(test_scores.mean()),
                "std_test_acc": float(test_scores.std(ddof=1))
                if len(test_scores) > 1
                else 0.0,
                "min_test_acc": float(test_scores.min()),
                "max_test_acc": float(test_scores.max()),
            }
        )

    if not aggregate_rows:
        return
    with aggregate_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(aggregate_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)


def _run_one_combo_fixed_split(
    elder_base,
    combo: dict,
    combo_total: int,
    raw_records,
    params: dict,
    *,
    result_dir: Path,
    epochs: int,
    model_seed: int,
    split_seed: int,
) -> dict:
    """Compose existing ElderAL steps without replacing any source function."""
    output_dir = result_dir / (
        f"{combo['combo_id']}+{combo['combo_name']}+csitime"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_steps = combo["pipeline_steps"]
    start_time = time.time()

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(elder_base.Tee(sys.stdout, log_file)):
            try:
                print("\n" + "=" * 80)
                print(
                    f"Ablation {combo['combo_index']}/{combo_total} | "
                    f"combo_id={combo['combo_id']} | model=csitime"
                )
                print(
                    f"model_seed={model_seed} | fixed_split_seed={split_seed}"
                )
                print(f"combo_name: {combo['combo_name']}")
                print(f"output: {output_dir}")
                print(json.dumps(pipeline_steps, ensure_ascii=False, indent=2))
                print("=" * 80)

                # Preprocessing is deterministic, but initialise it from the
                # fixed split seed in case a future source step uses randomness.
                elder_base.set_seed(split_seed)
                processed_data, labels, groups, unique_labels = elder_base.process_data(
                    raw_records,
                    pipeline_steps,
                    elder_base.PADDING_LENGTH,
                )
                split = elder_base.split_data(
                    processed_data,
                    labels,
                    groups,
                    pipeline_steps,
                    elder_base.TEST_SPLIT,
                    elder_base.VAL_SPLIT,
                    split_seed,
                )

                # Only model initialisation/training randomness varies here.
                elder_base.set_seed(model_seed)
                batch_size = (
                    elder_base.BATCH_SIZE
                    if elder_base.BATCH_SIZE is not None
                    else params.get("batch", 32)
                )
                loaders = elder_base.build_loaders(
                    split,
                    pipeline_steps,
                    batch_size,
                )
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"model input shape: {input_shape}")

                model, device = elder_base.create_registered_model(
                    "csitime",
                    len(unique_labels),
                    input_shape,
                )
                checkpoint_path = elder_base.train_registered_model(
                    model,
                    device,
                    loaders,
                    params,
                    output_dir,
                    elder_base.LEARNING_RATE,
                    elder_base.WEIGHT_DECAY,
                    epochs,
                    elder_base.PADDING_LENGTH,
                    combo["combo_id"],
                    "csitime",
                )
                val_acc, test_acc = elder_base.evaluate_checkpoint(
                    model,
                    device,
                    loaders[2],
                    checkpoint_path,
                )
                duration_sec = time.time() - start_time
                print(
                    f"Completed | val={val_acc:.4f} | test={test_acc:.4f} | "
                    f"duration={duration_sec:.2f}s"
                )
                return {
                    "combo_index": combo["combo_index"],
                    "combo_id": combo["combo_id"],
                    "combo_name": combo["combo_name"],
                    "model": "csitime",
                    "status": "ok",
                    "best_val_acc": val_acc,
                    "test_acc": test_acc,
                    "denoise": combo["denoise"],
                    "outliers": combo["outliers"],
                    "normalize": combo["normalize"],
                    "interpolate": combo["interpolate"],
                    "pipeline_steps": elder_base.pipeline_steps_to_json(
                        pipeline_steps
                    ),
                    "output_dir": str(output_dir),
                    "duration_sec": f"{duration_sec:.2f}",
                    "error": "",
                }
            except Exception:
                duration_sec = time.time() - start_time
                print("Ablation failed:")
                traceback.print_exc()
                return {
                    "combo_index": combo["combo_index"],
                    "combo_id": combo["combo_id"],
                    "combo_name": combo["combo_name"],
                    "model": "csitime",
                    "status": "failed",
                    "best_val_acc": "",
                    "test_acc": "",
                    "denoise": combo["denoise"],
                    "outliers": combo["outliers"],
                    "normalize": combo["normalize"],
                    "interpolate": combo["interpolate"],
                    "pipeline_steps": elder_base.pipeline_steps_to_json(
                        pipeline_steps
                    ),
                    "output_dir": str(output_dir),
                    "duration_sec": f"{duration_sec:.2f}",
                    "error": traceback.format_exc().splitlines()[-1],
                }


def run_csitime_ablation(
    run_name: str,
    combinations: Iterable[dict],
    *,
    epochs: int = 20,
    model_seeds: Iterable[int] = (42, 49, 514),
    split_seed: int = 42,
) -> Path:
    """Run independent combinations through the existing ElderAL trainer."""
    # The repository's training module imports PyTorch and is therefore loaded
    # only for an actual training run, not for config inspection/diagnostics.
    import full_test_elder as elder_base

    combinations = list(combinations)
    model_seeds = list(dict.fromkeys(int(seed) for seed in model_seeds))
    if not combinations:
        raise ValueError("at least one combination is required")
    if not model_seeds:
        raise ValueError("at least one model seed is required")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    # A seed suffix is part of the identity so resume logic cannot confuse
    # repeated model initialisations with an already completed run.
    seeded_combinations = []
    for model_seed in model_seeds:
        for template in combinations:
            combo = copy.deepcopy(template)
            combo["combo_index"] = len(seeded_combinations) + 1
            combo["combo_id"] = f"{template['combo_id']}_seed{model_seed}"
            combo["combo_name"] = (
                f"{template['combo_name']}+modelseed{model_seed}"
            )
            seeded_combinations.append((model_seed, combo))

    data_path = elder_data_path()
    result_dir = ABLATION_RESULT_ROOT / run_name
    summary_path = result_dir / f"{run_name}_csitime_summary.csv"

    if elder_base.MODEL_NAME != "csitime":
        raise RuntimeError("full_test_elder MODEL_NAME is no longer csitime")

    result_dir.mkdir(parents=True, exist_ok=True)
    completed = _successful_combo_ids(summary_path)

    print(f"ElderAL data: {data_path}")
    print(f"Ablation: {run_name}")
    print(f"CSI-Time epochs: {epochs}")
    print(f"Fixed data-split seed: {split_seed}")
    print(f"Model seeds: {model_seeds}")
    print(f"Results: {result_dir}")
    print(f"Completed: {len(completed)}/{len(seeded_combinations)}")

    params = elder_base.load_params(elder_base.DATASET_NAME)
    elder_base.set_seed(split_seed)
    raw_records = load_elder_records(data_path)
    failures = []

    for model_seed, combo in seeded_combinations:
        if combo["combo_id"] in completed:
            print(f"Skip completed: {combo['combo_id']}")
            continue

        print(
            f"Run {combo['combo_index']}/{len(seeded_combinations)} | "
            f"model_seed={model_seed} | split_seed={split_seed}"
        )
        try:
            row = _run_one_combo_fixed_split(
                elder_base,
                combo,
                len(seeded_combinations),
                raw_records,
                params,
                result_dir=result_dir,
                epochs=epochs,
                model_seed=model_seed,
                split_seed=split_seed,
            )
        finally:
            elder_base.clear_cuda_cache()

        _append_summary(summary_path, elder_base.SUMMARY_FIELDS, row)
        if row["status"] == "ok":
            completed.add(combo["combo_id"])
        else:
            failures.append(f"{combo['combo_id']}: {row['error']}")

    if failures:
        raise RuntimeError("Ablation failures:\n" + "\n".join(failures))

    aggregate_path = result_dir / f"{run_name}_csitime_aggregate.csv"
    _write_seed_aggregate(
        summary_path,
        aggregate_path,
        combinations,
        model_seeds,
    )
    print(f"Ablation complete. Summary: {summary_path}")
    print(f"Across-seed aggregate: {aggregate_path}")
    return summary_path
