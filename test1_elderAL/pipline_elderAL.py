"""
ElderAL 第一版完整 pipeline 测试脚本。

说明：
- 这是原始版本的 elderAL pipeline，不修改源码中的分组规则。
- 当前源码对 elderAL 的规则是 label = action_id，group = position_id。
- 因此本文件使用 GroupShuffleSplit 时，按位置 position_id 分组划分。

  "elderAL": {
    "batch": 32,
    "lr": 3e-4,
    "wd": 1e-3,
    "num_epochs": 20,
    "padding_length": 80
  },
  elif dataset in ('elderAL', 'zte'):
    label = int(res[2])   # action_id
    group = int(res[1])   # position_id
    elderAL 的 group 被源码定义成了 position_id
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

import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from wsdp import readers
from wsdp.models import create_model, list_models
from wsdp.datasets import CSIDataset
from wsdp.utils import resize_csi_to_fixed_length, train_model
from configurable_processor_elderAL import ConfigurableProcessor


# ======================= 配置区 ========================
"""按源码默认 group 划分 + 预设算法组合。"""
data_path = "./data/elderAL"
dataset_name = "elderAL"
preset_name = "baseprocessor"
# pipeline_steps = apply_preset(preset_name)
pipeline_steps = {
    "calibrate": {"method": "linear"},  # BaseProcessor 内部对应 phase_calibration
    "denoise": {"method": "wavelet"},   # BaseProcessor 内部对应 wavelet_denoise_csi
}
model_name = "csimodel"
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result",
    f"{preset_name}+{model_name}",
)
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 20
padding_length = 80
test_split = 0.3
val_split = 0.5
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


def main():
    print("=" * 80)
    print("WSDP ElderAL 第一版完整 pipeline 测试")
    print("=" * 80)

    print(f"step 1，加数据：{data_path}")
    print("   说明: readers.load_data() 会根据 dataset='elderAL' 自动选择 elderALReader")
    print("   说明: 使用本地 SDP 源码默认解析，label=action_id, group=position_id")
    csi_data_list = readers.load_data(data_path, dataset_name)
    print(f"共加载 {len(csi_data_list)} 个 CSI 样本")

    print("step 2，处理数据")
    print(f"当前 pipeline：{pipeline_steps}")
    print(f"当前 model：{model_name}")
    processor = ConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )
    print(f"处理完成: {len(all_data)} 个样本")
    print(f"标签分布: {dict((x, all_labels.count(x)) for x in set(all_labels))}")
    print(f"分组分布: {dict((x, all_groups.count(x)) for x in set(all_groups))}")
    print(f"样本形状: {all_data[0].shape}")

    print(f"step 3: 长度归一化 (padding_length={padding_length})")
    processed_data = resize_csi_to_fixed_length(all_data, padding_length)
    processed_data = np.array(processed_data)
    labels = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"数据 shape：{processed_data.shape}")

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = np.array([label_map[label] for label in labels])
    num_classes = len(unique_labels)
    print(f"类别数：{num_classes}，原始标签：{unique_labels}")

    print("step 4，数据划分")
    n_groups = len(set(groups))
    print(f"组数一共有 {n_groups}")
    if n_groups < 3:
        print(f"只有 {n_groups} 个 group，使用最简单的随机划分替代 GroupShuffleSplit")
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            processed_data,
            zero_indexed_labels,
            test_size=test_split,
            random_state=seed,
        )
        test_data, valid_data, test_labels, valid_labels = train_test_split(
            temp_data,
            temp_labels,
            test_size=val_split,
            random_state=seed,
        )
    else:
        splitter_1 = GroupShuffleSplit(
            n_splits=1,
            test_size=test_split,
            random_state=seed,
        )
        train_idx, temp_idx = next(
            splitter_1.split(processed_data, zero_indexed_labels, groups=groups)
        )
        train_data = processed_data[train_idx]
        train_labels = zero_indexed_labels[train_idx]

        temp_data = processed_data[temp_idx]
        temp_labels = zero_indexed_labels[temp_idx]
        temp_groups = groups[temp_idx]

        if len(set(temp_groups)) < 2:
            print(
                f"temp 里只有 {len(set(temp_groups))} 个 group，"
                "第二次切分退回普通 train_test_split"
            )
            test_data, valid_data, test_labels, valid_labels = train_test_split(
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

    print(f"训练集 {len(train_data)} | 验证集 {len(valid_data)} | 测试集 {len(test_data)}")

    train_dataset = CSIDataset(train_data, train_labels)
    valid_dataset = CSIDataset(valid_data, valid_labels)
    test_dataset = CSIDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

    if num_classes < 2:
        print(f"\n当前数据集只有 {num_classes} 个类别，训练结果仅作演示用途。")
        print("提示：使用包含多个 action 的完整数据集可获得更有意义的分类评估。")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
    train_process_path = os.path.join(output_dir, "train_process.txt")
    train_log_file = open(train_process_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, train_log_file)

    try:
        print(f"step 6: 开始训练 {num_epochs} 轮")
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

        print("step 7: 测试集评估")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
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
        print("pipeline 演示完成")
        print("=" * 60)

        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
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
