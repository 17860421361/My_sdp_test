import os
import sys
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# 当前脚本位于 test_gait 目录下，项目根目录是它的上一级。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 优先使用仓库中的 WSDP/SDP 源码，避免误导入环境里已安装的 wsdp 包。
LOCAL_WSDP_SRC = os.path.join(
    PROJECT_ROOT,
    "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main",
    "src",
)
if not os.path.isdir(os.path.join(LOCAL_WSDP_SRC, "wsdp")):
    raise FileNotFoundError(f"Local WSDP source not found: {LOCAL_WSDP_SRC}")
if LOCAL_WSDP_SRC not in sys.path:
    sys.path.insert(0, LOCAL_WSDP_SRC)

# sys.path 配置完成后再导入 wsdp，确保这里使用的是本地源码。
from wsdp import readers
from wsdp.algorithms import execute_pipeline, apply_preset
from wsdp.processors.base_processor import _parse_file_info_from_filename, _selector


class ConfigurableProcessor:
    """自定义算法的 pipeline，自定义预处理流程。"""

    def __init__(self, pipeline_steps):
        # 保存外部传入的算法流程，后续每个样本都会按这个流程处理。
        self.pipeline_steps = pipeline_steps

    def process(self, data_list, **kwargs):
        # dataset 用于解析文件名、选择标签/分组，以及传给部分算法。
        dataset = kwargs.get("dataset", "")
        all_data, all_labels, all_groups = [], [], []
        # 多进程 worker 只处理单个样本，这里用 partial 固定公共参数。
        worker_func = partial(
            _process_single_csi_configurable,
            dataset=dataset,
            pipeline_steps=self.pipeline_steps,
        )

        # 并行处理所有 CSI 样本；返回 None 的样本代表无效或帧数不足，会被过滤。
        with ProcessPoolExecutor(max_workers=16) as executor:
            result = executor.map(worker_func, data_list)
            for csi, label, group in result:
                if csi is not None:
                    all_data.append(csi)
                    all_labels.append(label)
                    all_groups.append(group)
        return all_data, all_labels, all_groups


def _process_single_csi_configurable(csi_data, dataset, pipeline_steps):
    """支持配置算法的单文件处理。"""

    # 兼容 CSIData 中 file_name / filename 两种字段命名。
    file_name = getattr(csi_data, "file_name", None)
    if file_name is None:
        file_name = getattr(csi_data, "filename", None)
    if file_name is None:
        raise AttributeError("CSIData object has neither 'file_name' nor 'filename'")

    # 从本地 SDP 源码解析标签和分组。gait 当前为 label=user_id, group=track*100+receiver。
    res = _parse_file_info_from_filename(file_name, dataset)
    label, group = _selector(res, dataset)

    # 按时间戳排序后，将每一帧的 CSI 矩阵堆叠成时间序列。
    sorted_frames = sorted(csi_data.frames, key=lambda f: f.timestamp)
    frame_tensors = [f.csi_array for f in sorted_frames]

    # 没有有效帧的样本直接跳过。
    if not frame_tensors:
        return None, None, None

    # 常见形状为 (T, F, A)：时间点、子载波、天线/链路维度。
    whole_csi = np.stack(frame_tensors, axis=0)

    # 单天线数据可能是 (T, F)，补一个天线维度变成 (T, F, 1)。
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    # 只有 1 个时间点无法做时序预处理，跳过。
    if whole_csi.shape[0] < 2:
        return None, None, None

    cleaned_csi = execute_pipeline(whole_csi, pipeline_steps)
    return cleaned_csi, label, group


if __name__ == "__main__":
    # Gait 数据集默认路径和 WSDP 内部识别的数据集名称。
    input_path = os.path.join(PROJECT_ROOT, "data", "Gait_Dataset", "CSI_Gait")
    dataset_name = "gait"

    if not os.path.isdir(input_path):
        print(f"示例数据路径不存在: {input_path}")
        print("请将 input_path 修改为实际的数据目录后重新运行")
        raise SystemExit(1)

    # 使用预设算法流程；也可以替换成自定义 pipeline_steps 字典。
    pipeline_steps = apply_preset("fast")

    print(f"正在加载数据：{input_path}......")

    csi_data_list = readers.load_data(input_path, dataset_name)
    print(f"共加载 {len(csi_data_list)} 个CSI数据样本")

    processor = ConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=dataset_name,
    )

    print("处理完成:")
    print("  gait 语义: label=user_id, group=track_id*100+receiver_id")
    print(f"  数据样本数: {len(all_data)}")
    print(f"  标签样本数: {len(all_labels)}")
    print(f"  分组样本数: {len(all_groups)}")
    if all_data:
        print(f"  单个样本形状: {all_data[0].shape}")

"""
共加载 22497 个CSI数据样本
处理完成:
  数据样本数: 22497
  标签样本数: 22497
  分组样本数: 22497
  单个样本形状: (8293, 30, 3)
"""
