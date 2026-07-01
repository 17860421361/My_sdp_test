"""
XRF55 完整 pipeline 测试脚本。

源码中 xrf55 的关键要求：
- dataset_name 必须是 "xrf55"，这样 readers.load_data() 才会选择 XrfReader。
- 当前 .npy 文件名格式为 user_action_trial.npy，例如 03_20_08.npy。
- 本测试脚本只读取按 user_id 排序后的前 3 个用户，后续用户不进入实验。
- 当前本地 SDP 源码中 xrf55 语义为 label=action_id, group=repetition_id。
- 数据划分直接使用源码 core._create_data_split():
  repetition 1-12 -> train, 13-16 -> valid, 17-20 -> test。
- XrfReader 会把每个 .npy reshape 为 (3, 30, 3, 1000)，再按 3 个 Rx 拆成
  3 个 CSIData 样本，所以前 3 个用户的 3300 个文件会变成 9900 个 CSI 样本。
- 单个样本形状通常是 (1000, 30, 3)，因此 padding_length 默认设为 1000。
  "xrf55": {
    "batch": 32,
    "lr": 3e-4,
    "wd": 1e-3,
    "num_epochs": 20,
    "padding_length": 1000
  },
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from wsdp.models import create_model, list_models
from wsdp.datasets import CSIDataset
from wsdp.utils import resize_csi_to_fixed_length, train_model
from wsdp.core import _create_data_split
from wsdp.processors import ConfigurableProcessor
from configurable_processor_xrf55 import load_xrf55_first_users


# ======================= 配置区 ========================
data_path = "./data/xrf55"
dataset_name = "xrf55"
preset_name = "baseprocessor"
# pipeline_steps = apply_preset(preset_name)
# BaseProcessor 等价流程，概念上是这个顺序
pipeline_steps = {
    # "calibrate": {"method": "linear"},  # phase_calibration
    "denoise": {"method": "wavelet"},   # wavelet_denoise_csi
}

model_name = "csimodel"
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result",
    f"repetition+{preset_name}+{model_name}",
)
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 20
padding_length = 1000
test_split = 0.3
val_split = 0.5
seed = 42
device_name = "cuda:0"
max_users_to_load = 3


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
    print("XRF55 WSDP 完整 pipeline 测试")
    print("=" * 80)

    print(f"step 1，加载数据：{data_path}")
    print("   说明: dataset='xrf55' 会自动选择 XrfReader")
    print("   说明: 每个 .npy 会按 3 个 Rx 拆成 3 个 CSIData 样本")
    (
        csi_data_list,
        selected_users,
        selected_file_count,
        raw_file_count,
    ) = load_xrf55_first_users(
        data_path,
        max_users=max_users_to_load,
    )
    print(f"   当前 xrf55 文件数: {raw_file_count}")
    print(
        f"   只读取前 {max_users_to_load} 个用户 {selected_users}: "
        f"{selected_file_count} 个文件，理论 CSIData 数: {selected_file_count * 3}"
    )
    print(f"共加载 {len(csi_data_list)} 个 CSI 样本")

    print("step 2，处理数据")
    print(f"当前 pipeline：{pipeline_steps}")
    print(f"当前 model：{model_name}")
    processor = ConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )
    if not all_data:
        raise RuntimeError("没有得到有效样本，请检查 xrf55 数据路径和 reader 输出。")

    print(f"处理完成: {len(all_data)} 个样本")
    print(f"标签分布: {dict((x, all_labels.count(x)) for x in set(all_labels))}")
    print(f"分组分布: {dict((x, all_groups.count(x)) for x in set(all_groups))}")
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

    print("step 4，使用源码 XRF55 repetition 划分")
    print("   train: repetition 1-12")
    print("   valid: repetition 13-16")
    print("   test : repetition 17-20")
    (
        train_data,
        valid_data,
        test_data,
        train_labels,
        valid_labels,
        test_labels,
    ) = _create_data_split(
        processed_data,
        zero_indexed_labels,
        groups,
        test_split,
        val_split,
        seed,
        use_simple_split=False,
        dataset=dataset_name,
    )

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
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
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
        print(f"当前 pipeline：{pipeline_steps}")
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
        print("XRF55 pipeline 测试完成")
        print("=" * 60)

        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("XRF55 Loss Curve")
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
