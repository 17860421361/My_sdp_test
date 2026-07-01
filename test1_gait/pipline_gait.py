"""
Gait 完整 pipeline 测试脚本。

源码中 gait 的关键要求：
- dataset_name 必须是 "gait"，这样 readers.load_data() 才会选择 BfeeReader。
- Gait 文件名格式为 user{用户}-{轨迹}-{重复}-r{接收器}.dat，
  例如 user3-1-1-r1.dat。
- BaseProcessor 对 gait 的解析规则：
  label = user_id，也就是文件名里的第 1 个数字。
  group = track_id * 100 + receiver_id，也就是按轨迹 + 接收器条件划分。
- 因此数据划分按非用户条件分组，保持闭集 User ID 分类。
- BfeeReader 输出单帧形状通常为 (30, n_rx*n_tx)，堆叠后是 (T, 30, A)。

model_params.json 中 gait 默认参数：
  "gait": {
    "batch": 32,
    "lr": 3e-4,
    "wd": 1e-3,
    "num_epochs": 60,
    "padding_length": 1500
  }
"""

import os
import sys

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
from wsdp.datasets import CSIDataset
from wsdp.models import create_model, list_models
from wsdp.utils import resize_csi_to_fixed_length, train_model
from configurable_processor_gait import ConfigurableProcessor

# ======================= 配置区 ========================
data_path = os.path.join(PROJECT_ROOT, "data", "Gait_Dataset", "CSI_Gait")
dataset_name = "gait"
processor_name = "source_user_id+baseprocessor"
# pipeline_steps = apply_preset(preset_name)
pipeline_steps = {
    "calibrate": {"method": "linear"},  # BaseProcessor 内部对应 phase_calibration
    "denoise": {"method": "wavelet"},   # BaseProcessor 内部对应 wavelet_denoise_csi
}
model_name = "csimodel"
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result",
    f"{processor_name}+{model_name}",
)
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 60
padding_length = 1500
test_split = 0.3
val_split = 0.5
seed = 42
cuda_device_index = 1


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


def _distribution(values):
    """把标签或分组计数整理成稳定顺序，方便查看日志。"""

    return {int(x): int(values.count(x)) for x in sorted(set(values))}


def _safe_train_test_split(data, labels, test_size, random_state):
    """样本太少无法 stratify 时，自动退回普通随机划分。"""

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
    """优先按 user_id 分组划分；分组不足时退回普通随机划分。"""

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
    print("Gait WSDP 完整 pipeline 测试")
    print("=" * 80)

    print(f"step 1，加载数据：{data_path}")
    print("   说明: dataset='gait' 会自动选择 BfeeReader")
    print("   说明: 文件名解析为 user_id / track_id / repetition_id / receiver_id")
    print("   说明: 使用本地 SDP 源码解析，label=user_id, group=track_id*100+receiver_id")
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
    processor = ConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )
    if not all_data:
        raise RuntimeError("没有得到有效样本，请检查 gait 数据路径、文件格式和 reader 输出。")

    print(f"处理完成: {len(all_data)} 个样本")
    print(f"标签分布: {_distribution(all_labels)}")
    print(f"分组分布: {_distribution(all_groups)}")
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

    # WSDP 的 CSIDataset 会对输入取 np.abs()，因此复数 CSI 会转成幅值训练。
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = np.array([label_map[label] for label in labels])
    num_classes = len(unique_labels)
    print(f"类别数：{num_classes}，原始标签：{unique_labels}")

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
    if torch.cuda.is_available() and torch.cuda.device_count() > cuda_device_index:
        device = torch.device(f"cuda:{cuda_device_index}")
    else:
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
        print(f"当前 processor：{processor_name}")
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
        print("Gait pipeline 测试完成")
        print("=" * 60)

        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gait Loss Curve")
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
