"""
Widar/Wider pipeline v3.

This script does not modify WSDP source code. It keeps the Widar Gesture task,
BaseProcessor-equivalent preprocessing, CSIModel, and cuda:0 training setup, but
uses a cleaner leave-one-user-out evaluation:

- label = gesture_type
- held-out user is used only for final test
- train/valid are split inside the remaining users by non-user condition group
- condition group = position_id * 1000 + orientation_id * 100 + receiver_id

This separates model selection from the unseen-user test set.
"""

import csv
import gc
import inspect
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_WSDP_SRC = os.path.join(
    PROJECT_ROOT,
    "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main",
    "src",
)
if not os.path.isdir(os.path.join(LOCAL_WSDP_SRC, "wsdp")):
    raise FileNotFoundError(f"Local WSDP source not found: {LOCAL_WSDP_SRC}")
if LOCAL_WSDP_SRC not in sys.path:
    sys.path.insert(0, LOCAL_WSDP_SRC)

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from wsdp import readers
from wsdp.algorithms import execute_pipeline
from wsdp.datasets import CSIDataset
from wsdp.models import create_model, list_models
from wsdp.utils import resize_csi_to_fixed_length, train_model


_EXECUTE_PIPELINE_ACCEPTS_DATASET = (
    "dataset" in inspect.signature(execute_pipeline).parameters
)


# ======================= 配置区 ========================
data_path = os.path.join(PROJECT_ROOT, "data", "widar_common3")
dataset_name = "widar"
processor_name = "leave_one_user_v3+baseprocessor"
pipeline_steps = {
    "calibrate": {"method": "linear"},  # phase_calibration
    "denoise": {"method": "wavelet"},   # wavelet_denoise_csi
}
model_name = "csimodel"
output_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result",
    f"{processor_name}+{model_name}",
)
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 80
padding_length = 1500
inner_val_split = 0.2
seed = 42
cuda_device_index = 0


class Tee:
    """同时把训练日志输出到终端和文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class WidarLeaveOneUserProcessor:
    """Local processor for Widar gesture recognition with user metadata."""

    def __init__(self, pipeline_steps):
        self.pipeline_steps = pipeline_steps

    def process(self, data_list, **kwargs):
        dataset = kwargs.get("dataset", "")
        all_data, all_labels, all_users, all_groups = [], [], [], []
        worker_func = partial(
            _process_single_csi_leave_one_user,
            dataset=dataset,
            pipeline_steps=self.pipeline_steps,
        )

        with ProcessPoolExecutor(max_workers=16) as executor:
            result = executor.map(worker_func, data_list)
            for csi, label, user, group in result:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_users.append(user)
                    all_groups.append(group)
        return all_data, all_labels, all_users, all_groups


def _parse_widar_filename(file_name):
    base = os.path.splitext(os.path.basename(file_name))[0]
    match = re.search(
        r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)",
        base,
        re.IGNORECASE,
    )
    if not match:
        return None

    user_id = int(match.group(1))
    gesture_type = int(match.group(2))
    position_id = int(match.group(3))
    orientation_id = int(match.group(4))
    serial_id = int(match.group(5))
    receiver_id = int(match.group(6))
    return user_id, gesture_type, position_id, orientation_id, serial_id, receiver_id


def _process_single_csi_leave_one_user(csi_data, dataset, pipeline_steps):
    file_name = getattr(csi_data, "file_name", None)
    if file_name is None:
        file_name = getattr(csi_data, "filename", None)
    if file_name is None:
        raise AttributeError("CSIData object has neither 'file_name' nor 'filename'")

    parsed = _parse_widar_filename(file_name)
    if parsed is None:
        return None, None, None, None

    user_id, gesture_type, position_id, orientation_id, _serial_id, receiver_id = parsed
    label = gesture_type
    group = position_id * 1000 + orientation_id * 100 + receiver_id

    sorted_frames = sorted(csi_data.frames, key=lambda f: f.timestamp)
    frame_tensors = [f.csi_array for f in sorted_frames]
    if not frame_tensors:
        return None, None, None, None

    whole_csi = np.stack(frame_tensors, axis=0)
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    if whole_csi.shape[0] < 2:
        return None, None, None, None

    if _EXECUTE_PIPELINE_ACCEPTS_DATASET:
        cleaned_csi = execute_pipeline(whole_csi, pipeline_steps, dataset=dataset)
    else:
        cleaned_csi = execute_pipeline(whole_csi, pipeline_steps)
    return cleaned_csi, label, user_id, group


def _distribution(values):
    items = [int(x) for x in values]
    return {x: items.count(x) for x in sorted(set(items))}


def _short_distribution(values, limit=12):
    distribution = _distribution(values)
    items = list(distribution.items())
    if len(items) <= limit:
        return distribution
    return {
        "count": len(items),
        "first_items": dict(items[:limit]),
    }


def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _safe_train_test_split(indices, labels, test_size, random_state):
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(unique) > 1 and np.min(counts) >= 2 else None
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def make_fold_indices(labels, users, groups, heldout_user):
    test_idx = np.where(users == heldout_user)[0]
    trainval_idx = np.where(users != heldout_user)[0]
    trainval_groups = groups[trainval_idx]
    trainval_labels = labels[trainval_idx]

    if len(set(trainval_groups)) < 2:
        print("训练用户内部 group 不足，退回普通分层随机划分 train/valid")
        train_rel, valid_rel = _safe_train_test_split(
            np.arange(len(trainval_idx)),
            trainval_labels,
            test_size=inner_val_split,
            random_state=seed,
        )
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=inner_val_split,
            random_state=seed,
        )
        train_rel, valid_rel = next(
            splitter.split(
                trainval_idx,
                trainval_labels,
                groups=trainval_groups,
            )
        )

    train_idx = trainval_idx[train_rel]
    valid_idx = trainval_idx[valid_rel]
    return train_idx, valid_idx, test_idx


def make_loaders(processed_data, labels, train_idx, valid_idx, test_idx):
    train_dataset = CSIDataset(processed_data[train_idx], labels[train_idx])
    valid_dataset = CSIDataset(processed_data[valid_idx], labels[valid_idx])
    test_dataset = CSIDataset(processed_data[test_idx], labels[test_idx])

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return float(np.mean(np.array(all_targets) == np.array(all_preds)))


def save_loss_curve(history, output_dir, title):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()


def run_fold(
    heldout_user,
    processed_data,
    labels,
    users,
    groups,
    num_classes,
    input_shape,
    device,
):
    cleanup_cuda()
    fold_output_dir = os.path.join(output_root, f"test_user{int(heldout_user)}")
    os.makedirs(fold_output_dir, exist_ok=True)

    checkpoint_path = os.path.join(fold_output_dir, "best_checkpoint.pth")
    train_process_path = os.path.join(fold_output_dir, "train_process.txt")
    train_log_file = open(train_process_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, train_log_file)

    model = None
    history = None
    checkpoint = None

    try:
        print("=" * 80)
        print(f"Widar/Wider leave-one-user-out fold: test_user={int(heldout_user)}")
        print("=" * 80)
        print(f"当前 processor：{processor_name}")
        print(f"当前 pipeline_steps流程：{pipeline_steps}")
        print(f"当前 model：{model_name}")
        print(f"保存目录：{fold_output_dir}")

        train_idx, valid_idx, test_idx = make_fold_indices(
            labels,
            users,
            groups,
            heldout_user,
        )

        train_users = sorted(set(users[train_idx].tolist()))
        valid_users = sorted(set(users[valid_idx].tolist()))
        test_users = sorted(set(users[test_idx].tolist()))
        train_groups = set(groups[train_idx].tolist())
        valid_groups = set(groups[valid_idx].tolist())

        if int(heldout_user) in train_users or int(heldout_user) in valid_users:
            raise RuntimeError("held-out user 泄漏进 train/valid，请检查划分逻辑。")
        if train_groups & valid_groups:
            raise RuntimeError("train/valid condition group 有重叠，请检查划分逻辑。")

        print(f"训练用户: {train_users}")
        print(f"验证用户: {valid_users}")
        print(f"测试用户: {test_users}")
        print(f"训练集 {len(train_idx)} | 验证集 {len(valid_idx)} | 测试集 {len(test_idx)}")
        print(f"训练标签分布: {_distribution(labels[train_idx].tolist())}")
        print(f"验证标签分布: {_distribution(labels[valid_idx].tolist())}")
        print(f"测试标签分布: {_distribution(labels[test_idx].tolist())}")
        print(f"训练 condition group 数: {len(train_groups)}")
        print(f"验证 condition group 数: {len(valid_groups)}")

        train_loader, valid_loader, test_loader = make_loaders(
            processed_data,
            labels,
            train_idx,
            valid_idx,
            test_idx,
        )

        print("step 5，创建模型")
        set_seed(seed)
        model = create_model(model_name, num_classes=num_classes, input_shape=input_shape)
        model = model.to(device)
        print(f"模型参数量 {sum(p.numel() for p in model.parameters())}")
        print(f"训练设备: {device}")

        print(f"step 6，开始训练 {num_epochs} 轮")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
        )

        history = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=valid_loader,
            num_epochs=num_epochs,
            device=device,
            checkpoint_path=checkpoint_path,
            padding_length=padding_length,
        )
        print(f"训练完成，最佳模型保存至: {checkpoint_path}")

        print("step 7，held-out user 测试集评估")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("未找到 checkpoint，使用当前最后一轮模型进行测试集评估")
        test_acc = evaluate(model, test_loader, device)
        best_val_acc = max(history["val_acc"]) if history["val_acc"] else np.nan
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"held-out user 测试准确率: {test_acc:.4f}")

        save_loss_curve(
            history,
            fold_output_dir,
            f"Widar v3 test user {int(heldout_user)} Loss Curve",
        )

        return {
            "heldout_user": int(heldout_user),
            "train_users": " ".join(str(x) for x in train_users),
            "valid_users": " ".join(str(x) for x in valid_users),
            "test_users": " ".join(str(x) for x in test_users),
            "train_size": len(train_idx),
            "valid_size": len(valid_idx),
            "test_size": len(test_idx),
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "output_dir": fold_output_dir,
        }
    finally:
        sys.stdout = original_stdout
        train_log_file.close()
        model = None
        history = None
        checkpoint = None
        cleanup_cuda()


def write_summary(rows):
    summary_path = os.path.join(output_root, "summary.csv")
    fieldnames = [
        "heldout_user",
        "train_users",
        "valid_users",
        "test_users",
        "train_size",
        "valid_size",
        "test_size",
        "best_val_acc",
        "test_acc",
        "output_dir",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary 保存至: {summary_path}")


def main():
    print("=" * 80)
    print("Widar/Wider WSDP pipeline v3: leave-one-user-out")
    print("=" * 80)

    print(f"step 1，加载数据：{data_path}")
    print("   说明: dataset='widar' 会自动选择 BfeeReader")
    print("   说明: 文件名解析为 user / gesture / position / orientation / serial / receiver")
    print("   说明: label=gesture, train/valid group=position*1000+orientation*100+receiver")
    raw_file_count = sum(
        1
        for _, _, files in os.walk(data_path)
        for name in files
        if name.endswith(".dat")
    )
    print(f"   当前 .dat 文件数: {raw_file_count}")

    csi_data_list = readers.load_data(data_path, dataset_name)
    print(f"共加载 {len(csi_data_list)} 个 CSI 样本")

    print("step 2，处理数据")
    print(f"当前 processor：{processor_name}")
    print(f"当前 pipeline_steps流程：{pipeline_steps}")
    print(f"当前 model：{model_name}")
    processor = WidarLeaveOneUserProcessor(pipeline_steps)
    all_data, all_labels, all_users, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )
    if not all_data:
        raise RuntimeError("没有得到有效样本，请检查 widar 数据路径、文件格式和 reader 输出。")

    print(f"处理完成: {len(all_data)} 个样本")
    print(f"用户分布: {_distribution(all_users)}")
    print(f"手势标签分布: {_distribution(all_labels)}")
    print(f"非用户 condition group 分布: {_short_distribution(all_groups)}")
    print(f"样本形状: {all_data[0].shape}")

    print(f"step 3，长度归一化 (padding_length={padding_length})")
    processed_data = resize_csi_to_fixed_length(
        all_data,
        target_length=padding_length,
    )
    processed_data = np.array(processed_data)
    users = np.array(all_users)
    groups = np.array(all_groups)
    labels = np.array(all_labels)
    print(f"数据 shape：{processed_data.shape}")

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = np.array([label_map[label] for label in labels])
    num_classes = len(unique_labels)
    input_shape = processed_data[0].shape
    print(f"类别数：{num_classes}，原始手势标签：{unique_labels}")

    print("step 4，leave-one-user-out 数据划分")
    heldout_users = sorted(set(users.tolist()))
    print(f"fold 顺序: {heldout_users}")

    print("step 5，准备设备")
    print("其他模型，例如：")
    for name, cat in sorted(list_models().items())[:5]:
        print(f"      [{cat}] {name}")
    print("      ... (更多模型请运行 list_models() 查看)")
    if torch.cuda.is_available() and torch.cuda.device_count() > cuda_device_index:
        device = torch.device(f"cuda:{cuda_device_index}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    os.makedirs(output_root, exist_ok=True)
    summary_rows = []
    for heldout_user in heldout_users:
        row = run_fold(
            heldout_user=heldout_user,
            processed_data=processed_data,
            labels=zero_indexed_labels,
            users=users,
            groups=groups,
            num_classes=num_classes,
            input_shape=input_shape,
            device=device,
        )
        summary_rows.append(row)
        write_summary(summary_rows)

    test_accs = [row["test_acc"] for row in summary_rows]
    val_accs = [row["best_val_acc"] for row in summary_rows]
    print("\n" + "=" * 80)
    print("Widar/Wider pipeline v3 全部 fold 完成")
    print(f"平均 best val acc: {float(np.mean(val_accs)):.2f}%")
    print(f"平均 held-out test acc: {float(np.mean(test_accs)):.4f}")
    print(f"输出目录: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    print("开始")
    main()
