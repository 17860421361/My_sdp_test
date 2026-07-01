"""
ElderAL pipeline V2。

修改说明：
- 不修改 WSDP 源码，也不修改原来的 pipline_elderAL.py。
- 原源码中 elderAL 的 group 是 position_id，本文件只在脚本内部把 group 改成 user_id。
- 数据划分逻辑改为先划分用户集合，再按用户取样本，保证 train/valid/test 用户互不重叠。
- 其余训练配置、模型、预处理 pipeline 和训练流程保持与 pipline_elderAL.py 基本一致。
- 输出目录加了 user_split_v2 后缀，避免覆盖原脚本已有结果。

  "elderAL": {
    "batch": 32,
    "lr": 3e-4,
    "wd": 1e-3,
    "num_epochs": 20,
    "padding_length": 80
  },
"""

import os
import sys
import inspect
from functools import partial
from concurrent.futures import ProcessPoolExecutor

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
from sklearn.model_selection import train_test_split
from wsdp import readers
from wsdp.models import create_model, list_models
from wsdp.datasets import CSIDataset
from wsdp.utils import resize_csi_to_fixed_length, train_model
from wsdp.algorithms import apply_preset, list_algorithms, list_presets, execute_pipeline
from wsdp.processors.base_processor import _parse_file_info_from_filename


_EXECUTE_PIPELINE_ACCEPTS_DATASET = (
    "dataset" in inspect.signature(execute_pipeline).parameters
)


class ElderALUserGroupConfigurableProcessor:
    """仅在本脚本内使用的 processor：预处理不变，但 elderAL 的 group 改为 user_id。"""

    def __init__(self, pipeline_steps):
        self.pipeline_steps = pipeline_steps

    def process(self, data_list, **kwargs):
        dataset = kwargs.get("dataset", "")
        all_data, all_labels, all_groups = [], [], []
        worker_func = partial(
            _process_single_csi_user_group,
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


def _process_single_csi_user_group(csi_data, dataset, pipeline_steps):
    """复制 configurable_processor 的处理逻辑，只把 elderAL 的 group 改成 user_id。"""

    file_name = getattr(csi_data, "file_name", None)
    if file_name is None:
        file_name = getattr(csi_data, "filename", None)
    if file_name is None:
        raise AttributeError("CSIData object has neither 'file_name' nor 'filename'")

    res = _parse_file_info_from_filename(file_name, dataset)
    if dataset != "elderAL":
        raise ValueError("pipline_elder_V2.py 只用于 elderAL 数据集")

    # elderAL 的解析结果是 (user_id, position_id, action_id, ...)
    # label 仍然是 action_id，但 group 改为 user_id，用于按用户划分。
    user_id = int(res[0])
    action_id = int(res[2])
    label = action_id
    group = user_id

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


def split_by_user_groups(processed_data, labels, groups):
    """先划分用户，再按用户取样本，保证 train/valid/test 用户不重叠。"""

    unique_groups = np.array(sorted(set(groups)))
    n_groups = len(unique_groups)
    print(f"用户组数一共有 {n_groups}，用户 ID: {unique_groups.tolist()}")

    if n_groups < 3:
        print(f"只有 {n_groups} 个用户，无法拆成互不重叠的 train/valid/test，退回随机划分")
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            processed_data,
            labels,
            test_size=test_split,
            random_state=seed,
        )
        test_data, valid_data, test_labels, valid_labels = train_test_split(
            temp_data,
            temp_labels,
            test_size=val_split,
            random_state=seed,
        )
        return train_data, valid_data, test_data, train_labels, valid_labels, test_labels

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    n_test_groups = max(1, int(round(n_groups * test_split)))
    test_groups = shuffled_groups[:n_test_groups]
    remaining_groups = shuffled_groups[n_test_groups:]

    n_val_groups = max(1, int(round(len(remaining_groups) * val_split)))
    if n_val_groups >= len(remaining_groups):
        n_val_groups = len(remaining_groups) - 1
    valid_groups = remaining_groups[:n_val_groups]
    train_groups = remaining_groups[n_val_groups:]

    train_mask = np.isin(groups, train_groups)
    valid_mask = np.isin(groups, valid_groups)
    test_mask = np.isin(groups, test_groups)

    print(
        "按用户划分结果："
        f"训练用户 {sorted(train_groups.tolist())} | "
        f"验证用户 {sorted(valid_groups.tolist())} | "
        f"测试用户 {sorted(test_groups.tolist())}"
    )

    train_data = processed_data[train_mask]
    train_labels = labels[train_mask]
    valid_data = processed_data[valid_mask]
    valid_labels = labels[valid_mask]
    test_data = processed_data[test_mask]
    test_labels = labels[test_mask]

    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels


# ======================= 配置区 ========================
""" 按用户划分 + 预设算法组合 """
data_path = "./data/elderAL"
dataset_name = "elderAL"
preset_name = "baseprocessor"
# pipeline_steps = apply_preset(preset_name)
pipeline_steps = {
    "calibrate": {"method": "linear"},  # BaseProcessor 内部对应 phase_calibration
    "denoise": {"method": "wavelet"},   # BaseProcessor 内部对应 wavelet_denoise_csi
}
model_name = "csitime"
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "result",
    f"v2+{preset_name}+{model_name}",
)
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 20
padding_length = 80
test_split = 0.3
val_split = 0.5
seed = 42


class Tee:
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
    print("WSDP ElderAL pipeline V2：按用户划分")
    print("=" * 80)

    print(f"step 1，加数据：{data_path}")
    print("   说明: readers.load_data() 会根据 dataset='elderAL' 自动选择 elderALReader")
    csi_data_list = readers.load_data(data_path, dataset_name)
    print(f"共加载 {len(csi_data_list)} 个 CSI 样本")

    print("step 2，处理数据")
    print(f"当前 pipeline：{pipeline_steps}")
    print(f"当前 model：{model_name}")
    print("当前 group：user_id，用于按用户划分")
    processor = ElderALUserGroupConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )
    print(f"处理完成: {len(all_data)} 个样本")
    print(f"标签分布: {dict((x, all_labels.count(x)) for x in set(all_labels))}")
    print(f"用户分组分布: {dict((x, all_groups.count(x)) for x in set(all_groups))}")
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
    (
        train_data,
        valid_data,
        test_data,
        train_labels,
        valid_labels,
        test_labels,
    ) = split_by_user_groups(processed_data, zero_indexed_labels, groups)

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

    print(f"step 6: 开始训练 {num_epochs} 轮")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
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
    print("pipeline V2 演示完成")
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
    sys.stdout = original_stdout
    train_log_file.close()


if __name__ == "__main__":
    print("开始")
    main()
