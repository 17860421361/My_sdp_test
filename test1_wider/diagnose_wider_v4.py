"""
Diagnostic script for Widar/Wider v4.

This script does not train a model and does not modify WSDP source code.
It reuses pipline_wider_v4.py preprocessing and checkpoint, then reports:

- original user-group split membership
- train/valid/test confusion matrices
- prediction distributions
- simple per user/gesture statistics on processed data

Run with the same conda environment used for training, for example:

    conda run -n sdp0425 python test_wider/diagnose_wider_v4.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

import pipline_wider_v4 as v4
from wsdp import readers
from wsdp.models import create_model
from wsdp.utils import resize_csi_to_fixed_length


MAX_EVAL_SAMPLES_PER_SPLIT = int(os.environ.get("WIDER_DIAG_MAX_EVAL", "300"))
EVAL_BATCH_SIZE = int(os.environ.get("WIDER_DIAG_BATCH_SIZE", "16"))


class Tee:
    def __init__(self, terminal_stream, file_stream):
        self.terminal_stream = terminal_stream
        self.file_stream = file_stream
        self.write_file = False

    def start_file_logging(self):
        self.write_file = True

    def write(self, data):
        self.terminal_stream.write(data)
        self.terminal_stream.flush()
        if self.write_file:
            self.file_stream.write(data)
            self.file_stream.flush()

    def flush(self):
        self.terminal_stream.flush()
        if self.write_file:
            self.file_stream.flush()


def _distribution(values):
    values = [int(x) for x in values]
    return {x: values.count(x) for x in sorted(set(values))}


def _safe_train_test_split_indices(indices, labels, test_size, random_state):
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(unique) > 1 and np.min(counts) >= 2 else None
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def split_indices_like_v4(labels, groups):
    n_groups = len(set(groups.tolist()))
    indices = np.arange(len(labels))
    print(f"group count: {n_groups}")

    if n_groups < 3:
        print("group count < 3, using random stratified split")
        train_idx, temp_idx = _safe_train_test_split_indices(
            indices,
            labels,
            test_size=v4.test_split,
            random_state=v4.seed,
        )
        test_idx, valid_idx = _safe_train_test_split_indices(
            temp_idx,
            labels[temp_idx],
            test_size=v4.val_split,
            random_state=v4.seed,
        )
        return train_idx, valid_idx, test_idx

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=v4.test_split,
        random_state=v4.seed,
    )
    train_idx, temp_idx = next(splitter_1.split(indices, labels, groups=groups))
    temp_groups = groups[temp_idx]

    if len(set(temp_groups.tolist())) < 2:
        print(
            f"temp has only {len(set(temp_groups.tolist()))} group, "
            "using random stratified split for valid/test"
        )
        test_idx, valid_idx = _safe_train_test_split_indices(
            temp_idx,
            labels[temp_idx],
            test_size=v4.val_split,
            random_state=v4.seed,
        )
    else:
        splitter_2 = GroupShuffleSplit(
            n_splits=1,
            test_size=v4.val_split,
            random_state=v4.seed,
        )
        test_rel, valid_rel = next(
            splitter_2.split(temp_idx, labels[temp_idx], groups=temp_groups)
        )
        test_idx = temp_idx[test_rel]
        valid_idx = temp_idx[valid_rel]

    return train_idx, valid_idx, test_idx


def confusion_matrix_np(targets, preds, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, preds):
        matrix[int(target), int(pred)] += 1
    return matrix


def _balanced_eval_subset(labels, max_samples, seed=42):
    if max_samples <= 0 or len(labels) <= max_samples:
        return np.arange(len(labels))

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    per_class = max(1, max_samples // len(classes))
    selected = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        take = min(per_class, len(cls_indices))
        selected.extend(rng.choice(cls_indices, size=take, replace=False).tolist())

    selected = np.array(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def evaluate_split(model, data, labels, device, split_name, num_classes):
    eval_idx = _balanced_eval_subset(labels, MAX_EVAL_SAMPLES_PER_SPLIT)
    original_count = len(labels)
    data = data[eval_idx]
    labels = labels[eval_idx]

    dataset = v4.ArrayCSIDataset(data, labels)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    all_preds, all_targets = [], []

    print(
        f"\n{split_name}: evaluating {len(labels)}/{original_count} samples "
        f"with batch_size={EVAL_BATCH_SIZE}"
    )
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist())
            if batch_idx % 5 == 0 or batch_idx == len(loader):
                print(f"{split_name}: evaluated batch {batch_idx}/{len(loader)}")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = float(np.mean(all_preds == all_targets))

    print("\n" + "=" * 80)
    print(f"{split_name} prediction diagnostics")
    print("=" * 80)
    print(f"accuracy: {acc:.4f}")
    print(f"target distribution: {_distribution(all_targets.tolist())}")
    print(f"prediction distribution: {_distribution(all_preds.tolist())}")
    print("confusion matrix, rows=true labels, cols=pred labels")
    print(confusion_matrix_np(all_targets, all_preds, num_classes))
    return acc


def per_user_gesture_stats(processed_data, labels, users):
    print("\n" + "=" * 80)
    print("Per user/gesture processed-data statistics")
    print("=" * 80)
    print("Columns: user, label, n, sample_mean_mean, sample_mean_std, sample_std_mean")

    sample_means = processed_data.mean(axis=(1, 2, 3))
    sample_stds = processed_data.std(axis=(1, 2, 3))

    rows = []
    for user in sorted(set(users.tolist())):
        for label in sorted(set(labels.tolist())):
            mask = (users == user) & (labels == label)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "user": int(user),
                    "label": int(label),
                    "n": int(mask.sum()),
                    "sample_mean_mean": float(sample_means[mask].mean()),
                    "sample_mean_std": float(sample_means[mask].std()),
                    "sample_std_mean": float(sample_stds[mask].mean()),
                }
            )

    for row in rows:
        print(
            f"user={row['user']:>2} label={row['label']:>2} "
            f"n={row['n']:>4} "
            f"mean_mu={row['sample_mean_mean']:+.4f} "
            f"mean_std={row['sample_mean_std']:.4f} "
            f"std_mu={row['sample_std_mean']:.4f}"
        )

    print("\nDistance between user/gesture centroids using sample means:")
    for label in sorted(set(labels.tolist())):
        label_rows = [row for row in rows if row["label"] == int(label)]
        for i, row_a in enumerate(label_rows):
            for row_b in label_rows[i + 1:]:
                dist = abs(row_a["sample_mean_mean"] - row_b["sample_mean_mean"])
                print(
                    f"label={int(label)} user{row_a['user']}-user{row_b['user']}: "
                    f"abs(mean_mu diff)={dist:.4f}"
                )


def main():
    os.makedirs(v4.output_dir, exist_ok=True)
    diagnose_path = os.path.join(v4.output_dir, "diagnose_wider_v4.txt")
    diagnose_log_file = open(diagnose_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    tee = Tee(original_stdout, diagnose_log_file)
    sys.stdout = tee

    try:
        _main(diagnose_path, tee)
    finally:
        sys.stdout = original_stdout
        diagnose_log_file.close()
        print(f"diagnose saved to: {diagnose_path}")


def _main(diagnose_path, tee):
    print("=" * 80)
    print("Widar/Wider v4 diagnostics")
    print("=" * 80)
    print(f"data_path: {v4.data_path}")
    print(f"checkpoint: {os.path.join(v4.output_dir, 'best_checkpoint.pth')}")
    print(f"diagnose output: {diagnose_path}")

    print("\nstep 1, load data")
    csi_data_list = readers.load_data(v4.data_path, v4.dataset_name)
    print(f"loaded samples: {len(csi_data_list)}")

    tee.start_file_logging()
    print("\nstep 2, process data with v4 processor logic")
    print("using sequential processing for diagnostic stability")
    all_data, all_labels, all_users = [], [], []
    for idx, csi_data in enumerate(csi_data_list, start=1):
        csi, label, group = v4._process_single_csi_zscore(
            csi_data,
            dataset=v4.dataset_name,
            pipeline_steps=v4.pipeline_steps,
        )
        if csi is not None:
            all_data.append(csi)
            all_labels.append(label)
            all_users.append(group)
        if idx % 500 == 0 or idx == len(csi_data_list):
            print(f"processed raw files: {idx}/{len(csi_data_list)}")
    if not all_data:
        raise RuntimeError("No valid samples after v4 processing")
    print(f"processed samples: {len(all_data)}")
    print(f"raw label distribution: {_distribution(all_labels)}")
    print(f"user/group distribution: {_distribution(all_users)}")
    print(f"single sample shape: {all_data[0].shape}")

    print("\nstep 3, resize")
    processed_data = resize_csi_to_fixed_length(
        all_data,
        target_length=v4.padding_length,
    )
    processed_data = np.array(processed_data, dtype=np.float32)
    raw_labels = np.array(all_labels)
    users = np.array(all_users)
    print(f"processed_data shape: {processed_data.shape}")
    print(
        "processed_data range: "
        f"min={float(np.min(processed_data)):.4f}, "
        f"max={float(np.max(processed_data)):.4f}, "
        f"mean={float(np.mean(processed_data)):.4f}, "
        f"std={float(np.std(processed_data)):.4f}"
    )

    unique_labels = sorted(set(raw_labels.tolist()))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in raw_labels])
    num_classes = len(unique_labels)
    print(f"label map: {label_map}")

    print("\nstep 4, reproduce v4 split")
    train_idx, valid_idx, test_idx = split_indices_like_v4(labels, users)
    for name, idx in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        print(
            f"{name}: n={len(idx)}, "
            f"users={sorted(set(users[idx].tolist()))}, "
            f"labels={_distribution(labels[idx].tolist())}"
        )

    print("\nstep 5, load v4 checkpoint")
    device = torch.device(
        f"cuda:{v4.cuda_device_index}"
        if torch.cuda.is_available() and torch.cuda.device_count() > v4.cuda_device_index
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = create_model(
        v4.model_name,
        num_classes=num_classes,
        input_shape=processed_data[0].shape,
    )
    checkpoint_path = os.path.join(v4.output_dir, "best_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"device: {device}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint best_val_acc: {checkpoint.get('best_val_acc')}")

    evaluate_split(
        model,
        processed_data[train_idx],
        labels[train_idx],
        device,
        "train",
        num_classes,
    )
    evaluate_split(
        model,
        processed_data[valid_idx],
        labels[valid_idx],
        device,
        "valid",
        num_classes,
    )
    evaluate_split(
        model,
        processed_data[test_idx],
        labels[test_idx],
        device,
        "test",
        num_classes,
    )

    per_user_gesture_stats(processed_data, labels, users)


if __name__ == "__main__":
    main()
