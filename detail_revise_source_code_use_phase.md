# 基于 dataset_name 自动选择幅度/相位输入的修改说明

## 1. 修改目标

本次修改把原先分散在 Gait、Widar 测试脚本中的 `USE_PHASE`、
`manual_phase_zscore` 和 `preserve_real_sign` 判断统一收到 WSDP 公共源码中。

最终规则如下：

| dataset_name | 模型输入策略 |
| --- | --- |
| `widar` | 自动使用幅度和相位 |
| `gait` | 自动使用幅度和相位 |
| `xrf55` | 只使用幅度，不增加相位通道 |
| `elderAL` | 只使用幅度，不增加相位通道 |

调用方只需要传递：

```python
dataset_name=DATASET_NAME
pipeline_steps=pipeline_steps
```

不再需要 `USE_PHASE` 或 `use_phase` 开关。

## 2. Widar/Gait 的最终数据流

### 2.1 pipeline 不包含 z-score

处理流程为：

```text
复数 CSI
→ ConfigurableProcessor/BaseProcessor
→ resize
→ split
→ CSIDataset 根据 dataset_name 判断为 Widar/Gait
→ amplitude = abs(CSI)
→ phase = angle(CSI)
→ concatenate([amplitude, phase], axis=-1)
→ 模型
```

假设原 CSI 形状为：

```text
(T, F, A)
```

进入模型的形状为：

```text
(T, F, 2A)
```

最后一维的前 `A` 个通道是幅度，后 `A` 个通道是相位。

### 2.2 pipeline 包含 z-score

源码 z-score 原本会执行：

```python
norm_amp * np.exp(1j * phase)
```

当 `norm_amp` 为负数时，这种复数表达会让相位额外偏移 `π`。后续再
执行 `abs()` 和 `angle()`，会同时丢失 z-score 负号并得到偏移后的相位。

现在的处理流程为：

```text
复数 CSI
→ 根据 dataset_name 判断为 Widar/Gait
→ 根据 pipeline_steps 判断包含 z-score
→ 暂时从本次 Processor 流程中移除 normalize
→ 执行降噪、相位校准、插值等其他步骤
→ 单独计算带正负号的 z-score 幅度
→ 直接拼接 [normalized_amplitude, phase]
→ resize
→ split
→ CSIDataset 识别为已经准备好的实数特征并直接保留
→ 模型
```

这样进入模型的是：

```text
[带正负号的 z-score 幅度, 正确的校准后相位]
```

z-score 在 `resize/padding` 之前完成，不会把 padding 值计算进均值和标准差。

## 3. 公共源码修改

### 3.1 `wsdp/dataset_policy.py`

文件：

```text
SDP/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/
src/wsdp/dataset_policy.py
```

新增：

```python
PHASE_AMPLITUDE_DATASETS = {"widar", "gait"}
```

新增两个公共判断函数：

```python
uses_phase_amplitude(dataset_name)
pipeline_uses_zscore(pipeline_steps)
```

数据集选择和 z-score 判断只在公共策略中定义一次，其他文件不再重复写条件。

### 3.2 `wsdp/algorithms/amplitude.py`

给 `normalize_amplitude()` 增加可选参数：

```python
return_phase_channels=False
```

默认值为 `False`，原有算法调用保持原行为。

当它为 `True` 时，返回：

```python
np.concatenate([norm_amp, phase], axis=-1)
```

而不是重新构造成复数，因此能够同时保留：

- z-score 幅度的正负号；
- 原始或校准后的真实相位。

### 3.3 `wsdp/processors/configurable_processor.py`

Processor 已经能收到 `dataset` 和 `pipeline_steps`，因此没有增加任何
`use_phase` 参数。

新增的自动判断是：

```python
phase_zscore = (
    uses_phase_amplitude(dataset)
    and pipeline_uses_zscore(pipeline_steps)
)
```

当判断成立时：

1. 本次执行时从 `effective_pipeline_steps` 中移除 `normalize`；
2. 执行其他 pipeline 算法；
3. 调用 `normalize_amplitude(..., return_phase_channels=True)`；
4. 返回实数形式的“带符号幅度 + 相位”通道。

XRF55 原有的“拆分之后使用训练集参数归一化”逻辑继续保留，没有改成
Widar/Gait 的处理方式。

### 3.4 `wsdp/datasets/CSIDataset.py`

删除了构造参数：

```python
use_phase
preserve_real_sign
```

新的公共接口是：

```python
CSIDataset(
    data_list,
    labels,
    dataset_name="",
    pipeline_steps=None,
)
```

内部行为：

- Widar/Gait + z-score：直接保留 Processor 已生成的实数幅度相位通道；
- Widar/Gait + 非 z-score：对复数 CSI 执行 `abs + angle + concatenate`；
- XRF55：不增加相位通道，并保留训练集归一化产生的实数正负号；
- ElderAL：不增加相位通道，保持原来的幅度路径；
- 其他未指定数据集：保持原来的幅度输入兼容行为。

如果 Widar/Gait + z-score 进入 `CSIDataset` 时仍然是复数，代码会直接报错。
这个检查可以防止错误地重新执行 `abs()`，避免问题被静默掩盖。

### 3.5 `wsdp/core.py`

`wsdp.core.pipeline()` 创建训练、验证和测试 Dataset 时，现在都会传递：

```python
dataset_name=dataset_name
pipeline_steps=resolved_pipeline_steps
```

模型输入形状改为从真正的 Dataset 输入读取：

```python
input_shape = tuple(train_dataset.data_list.shape[1:])
```

对于 Gait/Widar + z-score 这一条路径，train_data[0].shape 确实可以使用。
因为 z-score 已经在 ConfigurableProcessor 中完成拼接：
(T,30,3)
→ [norm_amp, phase]
→ (T,30,6)
所以：
train_data[0].shape
# (1500, 30, 6)
和：
tuple(train_dataset.data_list.shape[1:])
# (1500, 30, 6)
结果一样。
但原来的说法针对的是所有 pipeline。比如 Gait + fast 或 BaseProcessor：
Processor返回复数：(1500,30,3)
→ train_data[0].shape = (1500,30,3)
→ CSIDataset拼接幅度和相位
→ 模型输入 = (1500,30,6)
这时 train_data[0].shape 就错了。
所以准确说法应该改成：
对于 Gait/Widar + z-score，train_data[0].shape 与模型输入形状相同；但对于非 z-score 和 BaseProcessor，相位是在 CSIDataset 中加入的，两者形状不同。因此统一代码应从 train_dataset.data_list.shape[1:] 获取实际模型输入形状。

不能继续使用 `train_data[0].shape`，因为 Widar/Gait 加入相位后，最后一维
会从 `A` 变为 `2A`。

同时给 Widar/Gait 预处理缓存加入：

```text
automatic_amplitude_phase_v1
```

这会使旧的幅度缓存失效，防止新代码错误复用旧数据。

### 3.6 `wsdp/inference.py`

旧推理代码无条件执行：

```python
data = np.abs(data)
```

这会导致训练使用相位、推理却丢掉相位。

现在 `predict()` 和 `predict_single()` 增加：

```python
dataset_name=""
pipeline_steps=None
```

推理阶段通过 `CSIDataset` 使用与训练阶段完全相同的数据集策略。

对于 Widar/Gait z-score 模型，传入的推理数据应当是相同
`ConfigurableProcessor` 产生的预处理结果，并同时传入训练时的
`pipeline_steps`。

### 3.7 其他源码调用

以下源码调用同步传入了 `dataset_name`，并在需要的位置使用 Dataset 的
真实输入形状：

```text
src/wsdp/utils/hparam_search.py
scripts/benchmark_all_models.py
test_tools/run_full_pipeline.py
```

## 4. 四个数据集实验脚本修改

### 4.1 Gait

文件：

```text
SDP/test_gait/pipline_gait_steps.py
```

删除：

```text
USE_PHASE
manual_phase_zscore
effective_pipeline_steps
手动幅度均值/标准差计算
手动幅度相位拼接
preserve_real_sign
```

Processor 恢复为正常调用：

```python
ConfigurableProcessor(pipeline_steps)
```

Dataset 统一为：

```python
CSIDataset(
    data,
    labels,
    dataset_name=DATASET_NAME,
    pipeline_steps=pipeline_steps,
)
```

### 4.2 Widar

文件：

```text
SDP/test_wider/pipline_widar_steps.py
```

修改内容与 Gait 相同。所有幅度/相位判断都已从单独实验脚本删除。

### 4.3 ElderAL

文件：

```text
SDP/test_elderAL/pipline_elderAL_steps.py
```

删除局部 `preserve_real_sign` 判断，改为传递 `dataset_name` 和
`pipeline_steps`。ElderAL 不会增加相位通道。

### 4.4 XRF55

文件：

```text
SDP/test_xrf55/pipline_xrf55_steps.py
```

删除局部 `preserve_real_sign` 判断，改为传递 `dataset_name` 和
`pipeline_steps`。XRF55 不会增加相位通道，训练集归一化产生的实数
正负号仍然保留。

四个目录中的批量测试脚本复用对应 step 脚本，因此不需要再复制一套
相位判断逻辑。

### 4.5 四个 `pipline_*.py` 单组入口

以下四个直接调用 `wsdp.core.pipeline()` 的入口也已显式更新：

```text
SDP/test_elderAL/pipline_elderAL.py
SDP/test_gait/pipline_gait.py
SDP/test_wider/pipline_widar.py
SDP/test_xrf55/pipline_xrf55.py
```

它们现在通过公共 `uses_phase_amplitude(DATASET_NAME)` 输出实际输入策略，
没有增加本地相位开关：

```text
gait/widar → 幅度 + 相位
elderAL/xrf55 → 幅度
```

同时修正了两个单组入口的数据路径，使它们与 step 版本和实际数据目录一致：

```text
Gait:  sdp_dataset/Gait_Dataset/CSI_Gait
Widar: sdp_dataset/widar_common3
```

四个入口的数据路径均已实际检查为存在。

Gait/Widar 单组实验改用新的结果目录前缀：

```text
auto_amp_phase+{preset}+{model}
```

这样不会覆盖以前按幅度输入训练的同名 checkpoint。

### 4.6 四个 `full_test_presets*.py` 批量入口

以下四个批量入口均已修改：

```text
SDP/test_elderAL/full_test_presets_models_elderAL.py
SDP/test_gait/full_test_presets_models_gait.py
SDP/test_wider/full_test_presets_models_widar.py
SDP/test_xrf55/full_test_presets_models_xrf55.py
```

所有批量脚本现在统一从 DataLoader 中读取模型真正收到的形状：

```python
input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
```

并打印：

```python
print(f"模型实际输入形状: {input_shape}")
```

因此 Gait/Widar 的最后一维翻倍后，批量实验不会再使用处理前的旧形状创建模型。
ElderAL/XRF55 也使用同一个安全写法，但其通道数保持不变。

Gait/Widar 批量实验还更新了 `RUN_NAME`：

```text
Gait:  user_id_v3_auto_amp_phase
Widar: source_gesture_condition_auto_amp_phase
```

这一步是必要的：批量脚本会根据已有 summary 断点续跑。如果继续使用旧
`RUN_NAME`，以前的 amplitude-only 结果会让新幅度相位组合被误判为“已完成”
并直接跳过。ElderAL/XRF55 的输入表示没有变化，因此保持原 `RUN_NAME`。

## 5. 是否真的使用了相位

不是只修改了开关或日志。模型实际收到的 Tensor 最后一维已经包含独立相位通道。

使用真实数据文件验证结果：

```text
Widar:
raw_frames=1158
result_shape=(1158, 30, 6)
negative_amp=True
nonzero_phase=True
phase_range=[-3.1138, 3.1416]

Gait:
raw_frames=2306
result_shape=(2306, 30, 6)
negative_amp=True
nonzero_phase=True
phase_range=[-3.1227, 3.1416]
```

原数据最后一维是3，处理后最后一维是6：

```text
前3个通道：带正负号的 z-score 幅度
后3个通道：非零真实相位
```

实际多进程 `ConfigurableProcessor.process()` 也验证通过：

```text
samples=1
shape=(1158, 30, 6)
real=True
negative_amp=True
nonzero_phase=True
```

ElderAL 和 XRF55 真实文件验证结果：

```text
ElderAL: processed_shape=(32, 512, 3), model_shape=(32, 512, 3)
XRF55:   processed_shape=(1000, 30, 9), model_shape=(1000, 30, 9)
```

两者最后一维没有翻倍，证明没有加入相位通道。

## 6. 验证

新增了一个只针对本次数据表示的标准库测试：

```text
SDP/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/
tests/test_dataset_representation.py
```

测试覆盖：

1. 只有 Widar/Gait 自动选择幅度+相位；
2. `CSIDataset` 接口中不存在 `use_phase`；
3. Widar/Gait z-score 保留负幅度；
4. Widar/Gait z-score 保留真实相位；
5. 非 z-score 复数 CSI 自动拆成幅度和相位；
6. XRF55/ElderAL 不增加相位通道；
7. `normalize_amplitude()` 可以直接输出幅度相位通道；
8. 原 `normalize_amplitude()` 默认复数行为保持兼容。

执行结果：

```text
Ran 8 tests
OK
```

所有修改文件通过 `py_compile`，四个 step 脚本均可正常导入：

```text
elderAL import_ok
gait import_ok
widar import_ok
xrf55 import_ok
```

随后又对指定源码、四个数据集目录中的全部 Python 文件进行了完整语法解析：

```text
python_files_checked=123
syntax_errors=0
```

在忽略源码自带测试、只统计本次重点的 `src`、脚本和四个实验目录时，
最终再次检查了72个生产/实验 Python 文件：

```text
python_files_checked=72
syntax_errors=0
```

四个 `pipline_*.py`、四个 `pipline_*_steps.py` 和四个
`full_test_presets*.py` 共12个实验入口全部完成无训练导入检查：

```text
12/12 import_ok
```

对指定源码和四个数据集目录执行搜索后，没有残留以下 Python 逻辑：

```text
USE_PHASE
use_phase
manual_phase_zscore
preserve_real_sign
```

测试文件中只保留了一条反向断言，用来确认 `CSIDataset` 的公开参数确实
不包含 `use_phase`；生产源码和实验入口中不存在该开关。

另外运行了与本次修改直接相关的算法、registry、core 配置、inference 和
数据表示测试：

```text
103 passed
```

其中发现并修正了 `tests/test_core_configuration.py` 的 `DummyDataset`
仍使用旧构造签名的问题。该测试替身现在能够接收 `dataset_name` 和
`pipeline_steps`，并提供真实的 `data_list.shape` 给 core 输入形状检查。

## 7. 兼容性说明

### 7.1 Widar/Gait 旧 checkpoint

以前只使用幅度的模型输入形状是：

```text
(T, F, A)
```

现在自动使用幅度+相位后是：

```text
(T, F, 2A)
```

因此旧的 amplitude-only Widar/Gait checkpoint 不能直接当成新模型继续训练或
推理，需要重新训练。这是输入特征真正加入相位后的必然变化。

### 7.2 老版本诊断脚本

`SDP/test_gait/evaluate_old_checkpoint_on_current.py` 专门用于比较历史
amplitude-only checkpoint，因此显式使用空数据集策略保持旧输入形状。
它不属于新的训练路径，也没有重新引入 `use_phase`。

### 7.3 未传 dataset_name 的旧外部调用

`CSIDataset` 的 `dataset_name` 默认值是空字符串。旧代码如果只传
`data_list, labels`，仍然走历史幅度路径，不会因接口增加而立即报错。
新的标准训练和推理路径已经全部显式传递真实 `dataset_name`。

## 8. 最小改动原则

本次没有给四个数据集分别创建新的 Dataset，也没有在每个测试脚本中复制判断。

公共职责只有三处：

```text
dataset_policy     决定哪个数据集使用相位
ConfigurableProcessor 解决 Widar/Gait + z-score 的有符号幅度问题
CSIDataset         生成模型最终输入
```

测试脚本只负责传递 `dataset_name` 和 `pipeline_steps`，这是当前结构下能够同时
满足“全局统一、真正使用相位、保持 z-score 正负号、避免 padding 影响”的最小方案。
