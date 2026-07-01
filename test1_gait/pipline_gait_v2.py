"""
Gait User ID pipeline v2.

This script does not modify WSDP source code. It keeps the same reader,
preprocessing steps, model, and training settings as pipline_gait.py, but fixes
the gait task semantics locally:

- label = user_id
- group = track_id * 100 + receiver_id

The non-user group keeps train/valid/test as closed-set User ID splits while
holding out track/receiver conditions.
"""

import inspect
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# 只向 PyTorch 暴露物理 GPU 1；进程内对应逻辑设备 cuda:0。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_WSDP_SRC = os.path.join(
    PROJECT_ROOT,
    "SDP",
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
from wsdp.algorithms import apply_preset
from wsdp import readers
from wsdp.algorithms import execute_pipeline
from wsdp.datasets import CSIDataset
from wsdp.models import create_model, list_models
from wsdp.utils import resize_csi_to_fixed_length, train_model


_EXECUTE_PIPELINE_ACCEPTS_DATASET = (
    "dataset" in inspect.signature(execute_pipeline).parameters
)


# ======================= 配置区 ========================
data_path = os.path.join(
    PROJECT_ROOT,
    "sdp_dataset",
    "Gait_Dataset",
    "CSI_Gait",
)
dataset_name = "gait"
# processor_name = "gait_user_id_v2"

preset_name = "fast"
pipeline_steps = apply_preset(preset_name)
# BaseProcessor 等价流程，概念上是这个顺序
# pipeline_steps = {
#     "calibrate": {"method": "linear"},  # phase_calibration
#     "denoise": {"method": "wavelet"},   # wavelet_denoise_csi
# }

model_name = "mlpmodel"
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result",
    f"aaa_user_id_v2+{preset_name}+{model_name}",
)
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 60
padding_length = 1500
test_split = 0.3
val_split = 0.5
seed = 42


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


class GaitUserIdProcessor:
    """Local processor for the Gait User ID task."""

    def __init__(self, pipeline_steps):
        self.pipeline_steps = pipeline_steps

    def process(self, data_list, **kwargs):
        dataset = kwargs.get("dataset", "")
        all_data, all_labels, all_groups = [], [], []
        worker_func = partial(
            _process_single_csi_user_id,
            dataset=dataset,
            pipeline_steps=self.pipeline_steps,
        )

        with ProcessPoolExecutor(max_workers=16) as executor:
            result = executor.map(worker_func, data_list)
            for csi, label, group in result:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_groups.append(group)
        return all_data, all_labels, all_groups


def _parse_gait_filename(file_name):
    base = os.path.splitext(os.path.basename(file_name))[0]
    match = re.search(r"user(\d+)-(\d+)-(\d+)-r(\d+)", base, re.IGNORECASE)
    if not match:
        return None
    user_id = int(match.group(1))
    track_id = int(match.group(2))
    repetition_id = int(match.group(3))
    receiver_id = int(match.group(4))
    return user_id, track_id, repetition_id, receiver_id


def _process_single_csi_user_id(csi_data, dataset, pipeline_steps):
    file_name = getattr(csi_data, "file_name", None)
    if file_name is None:
        file_name = getattr(csi_data, "filename", None)
    if file_name is None:
        raise AttributeError("CSIData object has neither 'file_name' nor 'filename'")

    parsed = _parse_gait_filename(file_name)
    if parsed is None:
        return None, None, None

    user_id, track_id, _repetition_id, receiver_id = parsed
    label = user_id
    group = track_id * 100 + receiver_id

    sorted_frames = sorted(csi_data.frames, key=lambda f: f.timestamp)
    frame_tensors = [f.csi_array for f in sorted_frames]
    if not frame_tensors:
        return None, None, None

    whole_csi = np.stack(frame_tensors, axis=0)
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    if whole_csi.shape[0] < 2:
        return None, None, None

    if _EXECUTE_PIPELINE_ACCEPTS_DATASET:
        cleaned_csi = execute_pipeline(whole_csi, pipeline_steps, dataset=dataset)
    else:
        cleaned_csi = execute_pipeline(whole_csi, pipeline_steps)
    return cleaned_csi, label, group


def _distribution(values):
    return {int(x): int(values.count(x)) for x in sorted(set(values))}


def _safe_train_test_split(data, labels, test_size, random_state):
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(unique) > 1 and np.min(counts) >= 2 else None
    return train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _split_by_group_or_random(processed_data, labels, groups):
    n_groups = len(set(groups))
    print(f"组数一共有 {n_groups}")

    if n_groups < 3:
        print(f"只有 {n_groups} 个 group，使用普通随机划分替代 GroupShuffleSplit")
        train_data, temp_data, train_labels, temp_labels = _safe_train_test_split(
            processed_data,
            labels,
            test_size=test_split,
            random_state=seed,
        )
        test_data, valid_data, test_labels, valid_labels = _safe_train_test_split(
            temp_data,
            temp_labels,
            test_size=val_split,
            random_state=seed,
        )
        return train_data, valid_data, test_data, train_labels, valid_labels, test_labels

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_split,
        random_state=seed,
    )
    train_idx, temp_idx = next(splitter_1.split(processed_data, labels, groups=groups))
    train_data = processed_data[train_idx]
    train_labels = labels[train_idx]

    temp_data = processed_data[temp_idx]
    temp_labels = labels[temp_idx]
    temp_groups = groups[temp_idx]

    if len(set(temp_groups)) < 2:
        print(
            f"temp 里只有 {len(set(temp_groups))} 个 group，"
            "第二次切分退回普通 train_test_split"
        )
        test_data, valid_data, test_labels, valid_labels = _safe_train_test_split(
            temp_data,
            temp_labels,
            test_size=val_split,
            random_state=seed,
        )
    else:
        splitter_2 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_split,
            random_state=seed,
        )
        test_idx, val_idx = next(
            splitter_2.split(temp_data, temp_labels, groups=temp_groups)
        )
        test_data = temp_data[test_idx]
        test_labels = temp_labels[test_idx]
        valid_data = temp_data[val_idx]
        valid_labels = temp_labels[val_idx]

    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels


def main():
    print("=" * 80)
    print("Gait User ID WSDP pipeline v2")
    print("=" * 80)

    print(f"step 1，加载数据：{data_path}")
    print("   说明: dataset='gait' 会自动选择 BfeeReader")
    print("   说明: 文件名解析为 user_id / track_id / repetition_id / receiver_id")
    print("   说明: label=user_id, group=track_id*100+receiver_id")
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
    print(f"当前 processor：{preset_name}")
    print(f"当前 pipeline_steps流程：{pipeline_steps}")
    print(f"当前 model：{model_name}")
    processor = GaitUserIdProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )
    if not all_data:
        raise RuntimeError("没有得到有效样本，请检查 gait 数据路径、文件格式和 reader 输出。")

    print(f"处理完成: {len(all_data)} 个样本")
    print(f"用户标签分布: {_distribution(all_labels)}")
    print(f"非用户分组分布: {_distribution(all_groups)}")
    print(f"样本形状: {all_data[0].shape}")

    print(f"step 3，长度归一化 (padding_length={padding_length})")
    processed_data = resize_csi_to_fixed_length(
        all_data,
        target_length=padding_length,
    )
    processed_data = np.array(processed_data)
    labels = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"数据 shape：{processed_data.shape}")

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = np.array([label_map[label] for label in labels])
    num_classes = len(unique_labels)
    print(f"类别数：{num_classes}，原始用户标签：{unique_labels}")

    print("step 4，数据划分")
    (
        train_data,
        valid_data,
        test_data,
        train_labels,
        valid_labels,
        test_labels,
    ) = _split_by_group_or_random(processed_data, zero_indexed_labels, groups)

    print(f"训练集 {len(train_data)} | 验证集 {len(valid_data)} | 测试集 {len(test_data)}")
    print(f"训练标签分布: {_distribution(train_labels.tolist())}")
    print(f"验证标签分布: {_distribution(valid_labels.tolist())}")
    print(f"测试标签分布: {_distribution(test_labels.tolist())}")

    train_dataset = CSIDataset(train_data, train_labels)
    valid_dataset = CSIDataset(valid_data, valid_labels)
    test_dataset = CSIDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("step 5，创建模型")
    print(f"当前模型 {model_name}")
    print("其他模型，例如：")
    for name, cat in sorted(list_models().items())[:5]:
        print(f"      [{cat}] {name}")
    print("      ... (更多模型请运行 list_models() 查看)")

    input_shape = processed_data[0].shape
    model = create_model(model_name, num_classes=num_classes, input_shape=input_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型参数量 {sum(p.numel() for p in model.parameters())}")
    print(f"训练设备: {device}")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
    train_process_path = os.path.join(output_dir, "train_process.txt")
    train_log_file = open(train_process_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, train_log_file)

    try:
        print(f"step 6，开始训练 {num_epochs} 轮")
        print(f"当前 processor：{preset_name}")
        print(f"当前 pipeline_steps流程：{pipeline_steps}")
        print(f"当前 model：{model_name}")
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

        print("step 7，测试集评估")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("未找到 checkpoint，使用当前最后一轮模型进行测试集评估")
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                preds = torch.argmax(model(x), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = np.mean(np.array(all_targets) == np.array(all_preds))
        print(f"测试集准确率: {accuracy:.4f}")
        print("\n" + "=" * 60)
        print("Gait User ID pipeline v2 测试完成")
        print("=" * 60)

        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gait User ID Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    finally:
        sys.stdout = original_stdout
        train_log_file.close()


if __name__ == "__main__":
    print("开始")
    main()
