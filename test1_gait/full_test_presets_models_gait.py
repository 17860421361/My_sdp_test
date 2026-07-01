"""
Run all WSDP presets against all registered models on the Gait User ID task.

The batch-test flow follows test_xrf55/full_test_presets_models_xrf55.py.
The Gait-specific processing logic follows pipline_gait.py and the local SDP
source code:

- label = user_id
- group = track_id * 100 + receiver_id
"""

import gc
import os
import sys
import traceback

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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from wsdp import readers
from wsdp.algorithms import apply_preset, list_presets
from wsdp.datasets import CSIDataset
from wsdp.models import create_model, list_models
from wsdp.utils import resize_csi_to_fixed_length, train_model

from configurable_processor_gait import ConfigurableProcessor


# ======================= 配置区 ========================
data_path = os.path.join(PROJECT_ROOT, "data", "Gait_Dataset", "CSI_Gait")
dataset_name = "gait"
batch_size = 32
lr = 3e-4
weight_decay = 1e-3
num_epochs = 60
padding_length = 1500
test_split = 0.3
val_split = 0.5
seed = 42
cuda_device_index = 1
result_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
summary_path = os.path.join(result_root, "source_user_id_all_presets_models_summary.csv")
resume_from_summary = True
skip_model_names = {"visiontransformercsi", "mambacsi","graphneuralcsi"}
result_prefix = "source_user_id"


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


def make_loaders(processed_data, labels, groups):
    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = np.array([label_map[label] for label in labels])
    groups = np.array(groups)

    if len(set(groups)) < 3:
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            processed_data,
            zero_indexed_labels,
            test_size=test_split,
            random_state=seed,
            stratify=zero_indexed_labels,
        )
        test_data, valid_data, test_labels, valid_labels = train_test_split(
            temp_data,
            temp_labels,
            test_size=val_split,
            random_state=seed,
            stratify=temp_labels,
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
            test_data, valid_data, test_labels, valid_labels = train_test_split(
                temp_data,
                temp_labels,
                test_size=val_split,
                random_state=seed,
                stratify=temp_labels,
            )
        else:
            splitter_2 = GroupShuffleSplit(
                n_splits=1,
                test_size=val_split,
                random_state=seed,
            )
            test_idx, valid_idx = next(
                splitter_2.split(temp_data, temp_labels, groups=temp_groups)
            )
            test_data = temp_data[test_idx]
            test_labels = temp_labels[test_idx]
            valid_data = temp_data[valid_idx]
            valid_labels = temp_labels[valid_idx]

    train_dataset = CSIDataset(train_data, train_labels)
    valid_dataset = CSIDataset(valid_data, valid_labels)
    test_dataset = CSIDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "input_shape": processed_data[0].shape,
        "num_classes": len(unique_labels),
        "unique_labels": unique_labels,
        "split_sizes": (len(train_data), len(valid_data), len(test_data)),
    }


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


def save_loss_curve(history, output_dir):
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
    plt.close()


def write_summary(rows):
    os.makedirs(result_root, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def load_existing_summary():
    if not os.path.exists(summary_path):
        return [], set()

    summary = pd.read_csv(summary_path)
    rows = summary.to_dict("records")
    completed = set()
    for _, row in summary.iterrows():
        completed.add((str(row["preset"]), str(row["model"])))
    return rows, completed


def make_skip_row(preset_name, model_name):
    output_dir = os.path.join(result_root, f"{result_prefix}+{preset_name}+{model_name}")
    return {
        "preset": preset_name,
        "model": model_name,
        "status": "skipped",
        "best_val_acc": np.nan,
        "test_acc": np.nan,
        "output_dir": output_dir,
        "error": "skipped_by_config",
    }


def get_pending_models(preset_name, model_names, completed_combos):
    return [
        model_name
        for model_name in model_names
        if (preset_name, model_name) not in completed_combos
    ]


def run_one_model(preset_name, pipeline_steps, model_name, loaders, device):
    cleanup_cuda()

    output_dir = os.path.join(result_root, f"{result_prefix}+{preset_name}+{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
    train_process_path = os.path.join(output_dir, "train_process.txt")
    train_log_file = open(train_process_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, train_log_file)

    model = None
    criterion = None
    optimizer = None
    scheduler = None
    history = None
    checkpoint = None

    try:
        print("=" * 80)
        print(f"当前 preset：{preset_name}")
        print(f"当前 pipeline：{pipeline_steps}")
        print(f"当前 model：{model_name}")
        print(f"保存目录：{output_dir}")
        print(f"类别数：{loaders['num_classes']}，原始用户标签：{loaders['unique_labels']}")
        train_n, valid_n, test_n = loaders["split_sizes"]
        print(f"训练集 {train_n} | 验证集 {valid_n} | 测试集 {test_n}")

        print("step 5，创建模型")
        set_seed(seed)
        model = create_model(
            model_name,
            num_classes=loaders["num_classes"],
            input_shape=loaders["input_shape"],
        )
        model = model.to(device)
        print(f"模型参数量 {sum(p.numel() for p in model.parameters())}")
        print(f"训练设备: {device}")

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
            train_loader=loaders["train_loader"],
            val_loader=loaders["valid_loader"],
            num_epochs=num_epochs,
            device=device,
            checkpoint_path=checkpoint_path,
            padding_length=padding_length,
        )
        print(f"训练完成，最佳模型保存至: {checkpoint_path}")

        print("step 7: 测试集评估")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        accuracy = evaluate(model, loaders["test_loader"], device)
        print(f"测试集准确率: {accuracy:.4f}")
        print("\n" + "=" * 60)
        print("pipeline 演示完成")
        print("=" * 60)

        save_loss_curve(history, output_dir)

        return {
            "preset": preset_name,
            "model": model_name,
            "status": "ok",
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else np.nan,
            "test_acc": accuracy,
            "output_dir": output_dir,
            "error": "",
        }
    except Exception as exc:
        print("\n组合运行失败")
        print(f"preset={preset_name}, model={model_name}")
        print(f"error={repr(exc)}")
        traceback.print_exc()
        return {
            "preset": preset_name,
            "model": model_name,
            "status": "failed",
            "best_val_acc": np.nan,
            "test_acc": np.nan,
            "output_dir": output_dir,
            "error": repr(exc),
        }
    finally:
        sys.stdout = original_stdout
        train_log_file.close()
        model = None
        criterion = None
        optimizer = None
        scheduler = None
        history = None
        checkpoint = None
        cleanup_cuda()


def main():
    print("=" * 80)
    print("Gait User ID 全预设 + 全模型批量测试")
    print("=" * 80)
    print(f"使用 WSDP: {__import__('wsdp').__file__}")

    os.makedirs(result_root, exist_ok=True)

    print(f"step 1，加数据：{data_path}")
    csi_data_list = readers.load_data(data_path, dataset_name)
    print(f"共加载 {len(csi_data_list)} 个 CSI 样本")
    print("说明: label=user_id, group=track_id*100+receiver_id")

    preset_names = list(list_presets().keys())
    model_names = list(list_models().keys())
    print(f"预设数量：{len(preset_names)}，模型数量：{len(model_names)}")
    print(f"总组合数：{len(preset_names) * len(model_names)}")

    if torch.cuda.is_available() and torch.cuda.device_count() > cuda_device_index:
        device = torch.device(f"cuda:{cuda_device_index}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    if resume_from_summary:
        summary_rows, completed_combos = load_existing_summary()
        print(f"续跑模式: 已从 summary 读取 {len(completed_combos)} 个已记录组合")
    else:
        summary_rows, completed_combos = [], set()

    for preset_idx, preset_name in enumerate(preset_names, start=1):
        pending_models = get_pending_models(preset_name, model_names, completed_combos)

        print("\n" + "=" * 80)
        print(f"Preset [{preset_idx}/{len(preset_names)}]: {preset_name}")
        print(
            f"已记录 {len(model_names) - len(pending_models)}/{len(model_names)} "
            f"个模型组合"
        )

        if not pending_models:
            print(f"Preset 已完成，跳过数据处理: {preset_name}")
            print("=" * 80)
            continue

        skipped_now = [m for m in pending_models if m in skip_model_names]
        for model_name in skipped_now:
            combo_key = (preset_name, model_name)
            print(f"记录跳过配置中的模型: {preset_name}+{model_name}")
            row = make_skip_row(preset_name, model_name)
            summary_rows.append(row)
            completed_combos.add(combo_key)
            write_summary(summary_rows)

        pending_models = get_pending_models(preset_name, model_names, completed_combos)
        trainable_pending_models = [
            model_name for model_name in pending_models if model_name not in skip_model_names
        ]

        if not trainable_pending_models:
            print(f"Preset 剩余组合都已跳过，无需处理数据: {preset_name}")
            print("=" * 80)
            continue

        pipeline_steps = apply_preset(preset_name)
        print("\n" + "=" * 80)
        print(f"开始处理未完成 Preset [{preset_idx}/{len(preset_names)}]: {preset_name}")
        print(f"pipeline：{pipeline_steps}")
        print(f"待训练模型数：{len(trainable_pending_models)}")
        print("=" * 80)

        print("step 2，处理数据")
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

        print("step 4，数据划分")
        print("使用本地 SDP 源码 gait 语义: label=user_id, group=track_id*100+receiver_id")
        loaders = make_loaders(processed_data, labels, groups)

        for model_name in trainable_pending_models:
            model_idx = model_names.index(model_name) + 1
            print("\n" + "-" * 80)
            print(
                f"组合: preset {preset_idx}/{len(preset_names)} "
                f"model {model_idx}/{len(model_names)} -> {preset_name}+{model_name}"
            )
            combo_key = (preset_name, model_name)

            if combo_key in completed_combos:
                print(f"跳过已记录组合: {preset_name}+{model_name}")
                continue

            row = run_one_model(preset_name, pipeline_steps, model_name, loaders, device)
            summary_rows.append(row)
            completed_combos.add(combo_key)
            write_summary(summary_rows)

    print("\n" + "=" * 80)
    print("全部组合测试完成")
    print(f"汇总结果保存至: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    print("开始")
    main()
