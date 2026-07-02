# 以 `pipline_gait_steps.py` 为例的完整运行流程

## 1. 本文说明范围

本文以当前文件：

```text
SDP/test_gait/pipline_gait_steps.py
```

的默认配置为主线，详细说明：

- 程序从哪里开始运行；
- 每一步调用了哪个函数；
- 什么时候进入 WSDP 公共源码；
- 每个关键 `if` 条件为什么成立或不成立；
- 数据为什么一开始是复数；
- 数据什么时候变成实数；
- `phase_zscore` 在哪里触发；
- 为什么 z-score 后不再重新构造成复数；
- `CSIDataset` 如何决定直接保留数据还是执行 `abs + angle`；
- 数据最终如何进入模型、训练和测试。

当前默认配置是：

```python
DATASET_NAME = "gait"
PRESET_NAME = "high_quality"
PIPELINE_STEPS = None
MODEL_NAME = "mlpmodel"

NUM_EPOCHS = 60
PADDING_LENGTH = 1500
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 42
```

因此本文首先分析的实际组合是：

```text
Gait
+ high_quality
+ MLPModel
+ 自动使用幅度和相位
```

## 2. 默认运行流程总览

默认流程可以概括为：

```text
main()
→ run_experiment()
→ 设置随机种子
→ 把 high_quality 解析成 pipeline_steps 字典
→ load_raw_data()
→ BfeeReader 读取复数 CSI
→ process_data()
→ ConfigurableProcessor.process()
→ 每个样本进入 _process_single_csi_configurable()
→ 判断 phase_zscore=True
→ 暂时移除普通 normalize
→ execute_pipeline() 执行 butterworth + STC
→ normalize_amplitude(return_phase_channels=True)
→ 复数 (T,30,3) 变成实数 (T,30,6)
→ resize 到 (1500,30,6)
→ 标签和分组重新编号
→ split_data() 按 group 划分
→ build_loaders()
→ CSIDataset 判断 phase_zscore=True
→ 确认输入已经是实数
→ 直接保留 [z-score幅度, 相位]
→ 转成 torch.float32
→ 创建 MLPModel
→ 训练集训练、验证集选择最佳模型
→ 加载最佳 checkpoint
→ 测试集评估
```

## 3. 程序从哪里开始

文件最后是：

```python
if __name__ == "__main__":
    main()
```

直接运行：

```bash
python SDP/test_gait/pipline_gait_steps.py
```

时，当前文件的 `__name__` 等于 `"__main__"`，因此会调用：

```python
main()
```

`main()` 又调用：

```python
run_experiment(
    PRESET_NAME,
    MODEL_NAME,
    OUTPUT_DIR,
    PIPELINE_STEPS,
)
```

按照当前配置，实际参数是：

```text
preset_name = "high_quality"
model_name = "mlpmodel"
pipeline_steps_override = None
```

输出目录是：

```text
SDP/test_gait/result/self_design_test/
auto_amp_phase+high_quality+mlpmodel
```

## 4. 程序启动时的环境和路径

### 4.1 CUDA

文件导入 PyTorch 之前执行：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

因此当前程序只能看见物理 GPU 0。

后面创建设备时使用：

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

程序内部显示的 `cuda` 或 `cuda:0` 对应物理 GPU 0。

### 4.2 本地 WSDP 源码

程序构造：

```python
WSDP_SRC = (
    PROJECT_ROOT
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src"
)
```

随后执行：

```python
sys.path.insert(0, str(WSDP_SRC))
```

这意味着后面的：

```python
from wsdp import ...
```

优先导入项目目录中的 WSDP 源码，而不是其他环境里可能安装的同名包。

### 4.3 数据路径

Gait 数据目录是：

```text
sdp_dataset/Gait_Dataset/CSI_Gait
```

进入实验前：

```python
if not DATA_PATH.exists():
    raise FileNotFoundError(...)
```

如果目录不存在，程序在读取数据前就会终止。

## 5. `run_experiment()` 的准备阶段

### 5.1 创建输出目录

```python
output_dir.mkdir(parents=True, exist_ok=True)
```

用于保存：

```text
train_process.txt
best_checkpoint.pth
loss_curve.png
```

### 5.2 设置随机种子

```python
set_seed(SEED)
```

当前：

```text
SEED = 42
```

它设置：

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

这会固定 Python、NumPy 和 PyTorch 的主要随机过程。

### 5.3 解析 pipeline

调用：

```python
pipeline_steps = resolve_pipeline_steps(
    preset_name,
    pipeline_steps_override,
)
```

函数逻辑是：

```python
if pipeline_steps_override is not None:
    return pipeline_steps_override

if preset_name == "baseprocessor":
    return None

return apply_preset(preset_name)
```

当前条件判断如下：

```text
pipeline_steps_override is not None
→ None is not None
→ False
```

所以不使用自定义覆盖。

第二个判断：

```text
preset_name == "baseprocessor"
→ "high_quality" == "baseprocessor"
→ False
```

所以不使用 BaseProcessor。

最终执行：

```python
apply_preset("high_quality")
```

得到：

```python
pipeline_steps = {
    "denoise": {
        "method": "butterworth",
        "order": 5,
        "cutoff": 0.3,
    },
    "calibrate": {
        "method": "stc",
    },
    "normalize": {
        "method": "z-score",
    },
}
```

从这一刻开始，`pipeline_steps` 不再是 `None`，而是一个包含 z-score 的字典。

### 5.4 加载训练参数

```python
params = load_params("gait")
```

当脚本里的 `BATCH_SIZE`、`LEARNING_RATE`、`WEIGHT_DECAY` 为 `None` 时，会使用
配置文件中的默认值。

当前 batch 默认是：

```text
batch_size = 32
```

## 6. 第一步：`load_raw_data()`

调用：

```python
csi_data_list = load_raw_data()
```

内部执行：

```python
readers.load_data(
    str(DATA_PATH),
    DATASET_NAME,
)
```

此时：

```text
dataset = "gait"
```

### 6.1 为什么选择 `BfeeReader`

WSDP 的 Reader 注册表中：

```python
_READER_REGISTRY = {
    "widar": BfeeReader,
    "gait": BfeeReader,
    ...
}
```

所以 Gait 自动选择：

```python
BfeeReader
```

### 6.2 为什么原始 Gait 数据是复数

Intel IWL5300 的 CSI 文件同时保存每个 CSI 值的实部和虚部。

`BfeeReader` 首先创建：

```python
csi_array = np.zeros(
    (30, n_rx, n_tx),
    dtype=np.complex64,
)
```

读取二进制中的实部和虚部后执行：

```python
csi_array[...] = np.complex64(
    real8 + 1j * imag8
)
```

因此每个 CSI 值类似：

```text
3 + 4j
```

它可以表示为：

```text
幅度 = sqrt(3² + 4²) = 5
相位 = atan2(4, 3)
```

如果只保存实数，就无法同时保存原始幅度和相位。

每一帧最后被整理成：

```text
(F, A)
```

Gait 常见形状是：

```text
(30, 3)
```

其中：

```text
30 = 子载波数量
3  = 展平后的接收/发送天线组合数量
```

### 6.3 `csi_data_list` 的内容

`load_data()` 返回的是多个 `CSIData` 对象，而不是一个已经堆叠好的 NumPy 数组。

每个 `CSIData` 对象对应一个文件，内部包含：

```text
file_name
frames
```

每个 frame 包含：

```text
timestamp
csi_array
```

### 6.4 为什么 step 1 不写入训练日志

`load_raw_data()` 在：

```python
contextlib.redirect_stdout(...)
```

之前执行，所以读取文件时的输出只显示在终端，不写入 `train_process.txt`。

从第二步开始才同时写终端和日志文件。

## 7. 第二步：`process_data()`

进入日志重定向之后调用：

```python
processed_data, labels, groups, unique_labels = process_data(
    csi_data_list,
    pipeline_steps,
    1500,
)
```

### 7.1 选择哪个 Processor

代码是：

```python
processor = (
    BaseProcessor()
    if pipeline_steps is None
    else ConfigurableProcessor(pipeline_steps)
)
```

当前：

```text
pipeline_steps 是 high_quality 字典
pipeline_steps is None → False
```

因此创建：

```python
ConfigurableProcessor(pipeline_steps)
```

随后执行：

```python
processor.process(
    csi_data_list,
    dataset="gait",
)
```

## 8. `ConfigurableProcessor.process()` 做什么

`ConfigurableProcessor` 保存：

```python
self.pipeline_steps = pipeline_steps
```

然后从关键字参数中取出：

```python
dataset = kwargs.get("dataset", "")
```

当前：

```text
dataset = "gait"
```

它创建四个工作进程：

```python
ProcessPoolExecutor(max_workers=4)
```

每个 Gait 文件都会进入：

```python
_process_single_csi_configurable(
    csi_data,
    dataset="gait",
    pipeline_steps=high_quality字典,
)
```

工作进程返回：

```text
处理后的 CSI
标签
分组
```

如果处理结果不是 `None`，主进程才会加入：

```python
all_data
all_labels
all_groups
```

## 9. 单个 Gait 样本的详细处理

下面进入最关键的：

```python
_process_single_csi_configurable()
```

### 9.1 解析文件名

例如文件：

```text
user5-2-86-r5.dat
```

Gait 文件名被解析为：

```text
user_id       = 5
track_id      = 2
repetition_id = 86
receiver_id   = 5
```

`_selector()` 对 Gait 规定：

```python
label = user_id
group = track_id * 100 + receiver_id
```

所以该文件得到：

```text
label = 5
group = 2 * 100 + 5 = 205
```

`repetition_id=86` 会被解析，但当前 Gait 分组公式不使用它。

### 9.2 按时间排序 frame

```python
sorted_frames = sorted(
    csi_data.frames,
    key=lambda frame: frame.timestamp,
)
```

随后取出每一帧：

```python
frame_tensors = [
    frame.csi_array
    for frame in sorted_frames
]
```

### 9.3 空帧判断

```python
if not frame_tensors:
    return None, None, None
```

正常 Gait 文件有 frame，所以条件为 `False`，继续执行。

如果文件中没有有效 CSI frame，该样本会返回三个 `None`，最后不会加入训练数据。

### 9.4 堆叠时间维

```python
whole_csi = np.stack(
    frame_tensors,
    axis=0,
)
```

每帧是：

```text
(30, 3)
```

假设文件有 `T` 帧，堆叠后：

```text
(T, 30, 3)
```

数据类型仍是：

```text
complex64
```

因为 `np.stack()` 只增加时间维，不会删除虚部。

### 9.5 二维输入判断

```python
if whole_csi.ndim == 2:
    whole_csi = np.expand_dims(
        whole_csi,
        -1,
    )
```

正常 Gait：

```text
whole_csi.ndim = 3
```

因此条件为 `False`。

该分支是为了兼容单天线数据：

```text
(T, F)
→ (T, F, 1)
```

### 9.6 时间长度判断

```python
if whole_csi.shape[0] < 2:
    return None, None, None
```

正常文件至少有两个时间点，所以条件为 `False`。

只有零帧或一帧的样本无法进行正常时序处理，会被跳过。

## 10. `phase_zscore` 是如何触发的

代码是：

```python
phase_zscore = (
    uses_phase_amplitude(dataset)
    and pipeline_uses_zscore(pipeline_steps)
)
```

它由两个布尔条件组成。

### 10.1 第一个条件：是否使用幅度和相位

```python
uses_phase_amplitude("gait")
```

公共策略定义：

```python
PHASE_AMPLITUDE_DATASETS = {
    "widar",
    "gait",
}
```

内部判断：

```python
str(dataset).strip().lower()
in PHASE_AMPLITUDE_DATASETS
```

所以：

```text
"gait" in {"widar", "gait"}
→ True
```

### 10.2 第二个条件：pipeline 是否使用 z-score

```python
pipeline_uses_zscore(pipeline_steps)
```

内部判断：

```python
isinstance(pipeline_steps, dict)
and pipeline_steps
    .get("normalize", {})
    .get("method") == "z-score"
```

当前：

```text
pipeline_steps 是 dict
→ True

pipeline_steps["normalize"]["method"]
→ "z-score"

"z-score" == "z-score"
→ True
```

### 10.3 最终结果

```text
phase_zscore
= True and True
= True
```

它代表：

```text
当前数据集既需要幅度+相位，
当前 pipeline 又使用 z-score，
所以必须使用“带符号幅度+正确相位”的特殊表示。
```

## 11. 为什么先从 pipeline 中移除 normalize

代码先执行：

```python
effective_pipeline_steps = pipeline_steps
normalize_step = pipeline_steps.get(
    "normalize",
    {},
)
```

随后判断：

```python
if (
    phase_zscore
    or (
        dataset == "xrf55"
        and normalize_step.get("method")
            in {"z-score", "min-max"}
    )
):
```

当前：

```text
phase_zscore = True
```

所以整个 `if` 立即成立，不需要再依赖 XRF55 条件。

XRF55 子条件本身是：

```text
dataset == "xrf55"
→ "gait" == "xrf55"
→ False
```

但因为逻辑是：

```text
True or False
→ True
```

仍然进入 `if`。

执行：

```python
effective_pipeline_steps = {
    key: value
    for key, value in pipeline_steps.items()
    if key != "normalize"
}
```

原来的：

```python
{
    "denoise": ...,
    "calibrate": ...,
    "normalize": {"method": "z-score"},
}
```

变成：

```python
{
    "denoise": {
        "method": "butterworth",
        "order": 5,
        "cutoff": 0.3,
    },
    "calibrate": {
        "method": "stc",
    },
}
```

这里不是取消 z-score，而是暂时不让普通 pipeline 执行它。z-score 会在后面以正确
的幅度相位输出方式执行一次。

如果不移除 normalize，就会发生：

```text
普通 pipeline 先执行 z-score
→ norm_amp * exp(j*phase)
→ 负的 norm_amp 被编码成 phase+π
→ 后面无法同时恢复正确负号和原相位
```

## 12. `execute_pipeline()` 执行哪些算法

调用：

```python
cleaned_csi = execute_pipeline(
    whole_csi,
    effective_pipeline_steps,
    dataset="gait",
)
```

固定执行顺序是：

```python
[
    "denoise",
    "outliers",
    "calibrate",
    "normalize",
    "interpolate",
    "extract_features",
    "detect",
]
```

只有出现在 `effective_pipeline_steps` 中的类别才执行。

当前包含：

```text
denoise
calibrate
```

因此实际顺序是：

```text
复数 whole_csi
→ Butterworth 降噪
→ STC 相位校准
→ 复数 cleaned_csi
```

### 12.1 为什么降噪后通常仍然是复数

输入 `whole_csi` 是复数 CSI。降噪算法处理 CSI 信号，但没有执行：

```python
np.abs(...)
```

把它永久转换为纯幅度特征。

因此处理结果仍可保存实部和虚部，类型保持复数。

### 12.2 为什么 STC 校准后仍然是复数

STC 会取得幅度和校准后的相位，再重新构造：

```text
校准后复数 CSI
= amplitude * exp(j * calibrated_phase)
```

只要相位不是恒定的 `0` 或 `π`，结果就具有非零虚部，因此是复数数组。

### 12.3 为什么不会在每一步后把 Gait 转成实数

`execute_pipeline()` 每执行一个算法后都会调用：

```python
real_if_negligible_imaginary(
    result,
    dataset,
)
```

但这个函数只允许幅度主数据集转换：

```python
AMPLITUDE_PRIMARY_DATASETS = {
    "xrf55",
}
```

第一层判断是：

```python
if (
    not is_amplitude_primary_dataset(dataset)
    or not np.iscomplexobj(csi)
):
    return csi
```

对于 Gait：

```text
is_amplitude_primary_dataset("gait")
→ False

not False
→ True
```

因此直接返回原复数数据，不检查虚部是不是接近零。

原因是 Gait 需要保存相位，不能因为某一步恰好出现很小的虚部就提前丢掉复数表示。

## 13. 特殊 z-score 如何把复数转换成实数通道

`execute_pipeline()` 返回复数 `cleaned_csi` 后：

```python
if phase_zscore:
```

当前：

```text
phase_zscore = True
```

所以进入：

```python
cleaned_csi = normalize_amplitude(
    cleaned_csi,
    method="z-score",
    return_phase_channels=True,
)
```

### 13.1 计算幅度

```python
amplitude = np.abs(csi)
```

复数：

```text
a + bj
```

变成：

```text
sqrt(a² + b²)
```

`amplitude` 是实数数组，形状保持：

```text
(T, 30, 3)
```

### 13.2 计算 z-score

沿时间轴：

```python
mean = np.mean(
    amplitude,
    axis=0,
    keepdims=True,
)

std = np.std(
    amplitude,
    axis=0,
    keepdims=True,
)
```

均值和标准差形状是：

```text
(1, 30, 3)
```

每个“子载波 × 天线”位置使用自己的时间均值和标准差。

标准差过小时：

```python
std = np.where(
    std < 1e-10,
    1.0,
    std,
)
```

避免除零。

归一化：

```python
norm_amp = (
    amplitude - mean
) / std
```

`norm_amp` 是实数，而且可能为负数。

例如：

```text
某时刻幅度低于均值
→ amplitude - mean < 0
→ norm_amp < 0
```

### 13.3 提取相位

```python
phase = np.angle(csi)
```

`phase` 也是实数数组，形状是：

```text
(T, 30, 3)
```

数值通常位于：

```text
[-π, π]
```

### 13.4 直接拼接

因为：

```text
return_phase_channels = True
```

代码执行：

```python
return np.concatenate(
    [norm_amp, phase],
    axis=-1,
)
```

两个输入都是实数：

```text
norm_amp：实数
phase：实数
```

所以拼接结果也是实数。

形状变化：

```text
(T, 30, 3)
→ (T, 30, 6)
```

最后一维的含义是：

```text
索引 0、1、2：带正负号的 z-score 幅度
索引 3、4、5：对应的正确相位
```

### 13.5 为什么不再是复数

新结果没有执行：

```python
norm_amp * np.exp(1j * phase)
```

而是把两个物理量当成两个实数特征区域保存：

```text
[norm_amp, phase]
```

例如：

```text
[-3.0, 0.7]
```

代表：

```text
归一化幅度 = -3.0
相位 = 0.7 弧度
```

它不是：

```text
-3 * exp(0.7j)
```

所以不存在负幅度被解释成相位偏移 `π` 的问题。

## 14. 一个真实 Gait 文件的类型变化

使用真实文件：

```text
user5-2-86-r5.dat
```

实际验证结果：

```text
frames = 2306

读取后：
shape   = (2306, 30, 3)
dtype   = complex64
complex = True

high_quality处理后：
shape   = (2306, 30, 6)
dtype   = float32
complex = False

label = 5
group = 205

前三个幅度区域存在负数：
negative_amp = True

后三个相位区域存在非零相位：
phase_nonzero = True

相位范围约为：
[-3.08, 3.09]
```

这验证了：

```text
原始复数 CSI
→ 保留符号的实数 z-score 幅度
+ 实数相位
```

## 15. Processor 返回主进程

每个工作进程最终返回：

```python
return cleaned_csi, label, group
```

默认 high_quality 下：

```text
cleaned_csi：实数 (T,30,6)
label：原始 user_id
group：track_id*100+receiver_id
```

主进程只保存：

```python
if csi is not None:
```

的结果。

## 16. resize/padding

回到 `process_data()` 后：

```python
processed_data = resize_csi_to_fixed_length(
    all_data,
    target_length=1500,
)
```

### 16.1 长于 1500

```python
sample[:1500, :, :]
```

例如真实样本：

```text
(2306,30,6)
→ (1500,30,6)
```

### 16.2 短于 1500

在时间维末尾补零：

```text
(T,30,6)
→ (1500,30,6)
```

### 16.3 正好 1500

直接保留。

### 16.4 为什么 z-score 在 padding 前执行

默认 high_quality 的 z-score 已经在单样本 Processor 中完成，然后才 resize。

因此均值和标准差只根据真实时间帧计算，不会把为了补长度产生的零加入统计量。

## 17. 标签和分组重新编号

### 17.1 标签

```python
unique_labels = sorted(set(all_labels))
label_map = {
    label: idx
    for idx, label
    in enumerate(unique_labels)
}
```

假设原始用户标签：

```text
[1,2,3,4,5,6,7,8,9]
```

会映射成：

```text
1 → 0
2 → 1
...
9 → 8
```

因为交叉熵分类通常要求类别索引从 `0` 开始。

### 17.2 分组

原始 group 类似：

```text
101, 102, 103, ..., 406
```

同样映射成连续整数：

```text
0, 1, 2, ...
```

这里只改变编号，不改变“哪些样本属于同一组”的关系。

### 17.3 转成 NumPy 数组

```python
processed_data = np.asarray(processed_data)
labels = np.asarray(...)
groups = np.asarray(...)
```

默认 high_quality 下：

```text
processed_data：
(样本数,1500,30,6)
实数
```

## 18. 第三步：`split_data()`

调用：

```python
_create_data_split(
    processed_data,
    labels,
    groups,
    test_split=0.3,
    val_split=0.5,
    seed=42,
    use_simple_split=...,
    dataset="gait",
    pipeline_steps=high_quality字典,
)
```

### 18.1 是否使用普通随机划分

调用前计算：

```python
len(set(groups.tolist())) < 3
```

Gait 有多个 track/receiver 分组，通常远多于三个，因此：

```text
use_simple_split = False
```

所以使用：

```python
GroupShuffleSplit
```

而不是普通 `train_test_split`。

### 18.2 XRF55 专用分支

`_create_data_split()` 首先判断：

```python
if dataset == "xrf55":
```

当前：

```text
"gait" == "xrf55"
→ False
```

所以不会进入 XRF55 的固定 repetition 划分和训练集全局归一化逻辑。

### 18.3 第一次 group split

```python
GroupShuffleSplit(
    test_size=0.3,
    random_state=42,
)
```

按 group 把数据分为：

```text
约 70%：train
约 30%：temp
```

同一个 group 不会被拆到 train 和 temp 两边。

### 18.4 第二次 group split

再对 temp 执行：

```python
GroupShuffleSplit(
    test_size=0.5,
    random_state=42,
)
```

把暂存部分分成：

```text
约 15%：test
约 15%：val
```

实际样本比例可能因为每组样本数不同而略有偏差。

### 18.5 返回顺序

返回：

```text
train_data
val_data
test_data
train_labels
val_labels
test_labels
```

注意返回顺序是训练、验证、测试。

## 19. 第四步：构造 DataLoader

先决定 batch：

```python
batch_size = (
    BATCH_SIZE
    if BATCH_SIZE is not None
    else params.get("batch", 32)
)
```

当前：

```text
BATCH_SIZE = None
```

所以通常得到：

```text
batch_size = 32
```

训练、验证、测试分别调用：

```python
CSIDataset(
    data,
    labels,
    dataset_name="gait",
    pipeline_steps=high_quality字典,
)
```

## 20. `CSIDataset` 中每个条件如何判断

### 20.1 先转成 NumPy 数组

```python
data_array = np.asarray(data_list)
```

默认 high_quality 在 Processor 中已经完成特殊 z-score，所以此时：

```text
data_array 是实数
shape = (集合样本数,1500,30,6)
np.iscomplexobj(data_array) = False
```

### 20.2 判断是不是相位数据集

```python
phase_dataset = uses_phase_amplitude(
    dataset_name
)
```

当前：

```text
uses_phase_amplitude("gait")
→ True
```

所以：

```text
phase_dataset = True
```

### 20.3 再次判断 `phase_zscore`

```python
phase_zscore = (
    phase_dataset
    and pipeline_uses_zscore(pipeline_steps)
)
```

当前两个条件都为 `True`：

```text
phase_zscore = True
```

Processor 和 Dataset 都判断一次，但职责不同：

```text
Processor：
负责生成正确的实数 [norm_amp, phase]

CSIDataset：
负责确认这种数据已经生成，并避免重复 abs/angle
```

### 20.4 第一个 `if phase_zscore`

```python
if phase_zscore:
```

当前为 `True`，进入该分支。

里面先检查：

```python
if np.iscomplexobj(data_array):
    raise RuntimeError(...)
```

当前：

```text
np.iscomplexobj(data_array)
→ False
```

所以不会报错。

然后：

```python
data_list = data_array
```

直接保留：

```text
[带正负号的 z-score 幅度, 正确相位]
```

### 20.5 为什么设置了报错条件

如果 `phase_zscore=True`，但传进来的还是复数：

```text
np.iscomplexobj(data_array) = True
```

说明 `ConfigurableProcessor` 没有按规定生成实数幅度相位通道，或者调用方传错了
数据。

此时不能静默执行 `abs()`，否则会再次丢失 z-score 负号并改变负幅度对应的相位。

所以源码直接抛出：

```text
RuntimeError
```

让错误立即暴露。

### 20.6 为什么不会进入第二个 `elif`

第二个分支是：

```python
elif (
    phase_dataset
    and np.iscomplexobj(data_array)
):
```

Python 的 `if/elif` 结构只执行第一个成立的分支。

当前第一个：

```text
if phase_zscore
```

已经成立，所以不会继续判断这个 `elif`。

这正是为了避免对已经拼好的实数 `[norm_amp, phase]` 再执行一次：

```python
np.abs(...)
np.angle(...)
```

### 20.7 转成 PyTorch Tensor

```python
self.data_list = torch.from_numpy(
    data_list
).float()

self.labels = torch.from_numpy(
    labels
).long()
```

所以：

```text
数据 dtype：torch.float32
标签 dtype：torch.int64
```

训练 batch 的典型形状是：

```text
(32,1500,30,6)
```

## 21. DataLoader 的参数

当前创建：

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
```

### 21.1 `shuffle=True`

当前训练、验证、测试三个 loader 都设置为 `True`。

训练集打乱是正常训练做法。

验证集和测试集打乱只改变样本评估顺序，不改变最终总体 loss/accuracy，但通常可以
设置为 `False` 让评估顺序固定。

### 21.2 `num_workers=0`

表示 DataLoader 不额外创建数据加载子进程，数据由当前训练进程读取。

这不会改变数据内容，只影响数据加载方式和可能的速度。

## 22. 第五步：创建模型

输入形状从真正的 Dataset 中读取：

```python
tuple(
    loaders[0]
    .dataset
    .data_list
    .shape[1:]
)
```

默认得到：

```text
(1500,30,6)
```

这里不能继续只看 Processor 之前的原始 `(1500,30,3)`，因为模型真正接收的是
幅度和相位拼接后的 `6`。

调用：

```python
create_model(
    "mlpmodel",
    num_classes=9,
    input_shape=(1500,30,6),
)
```

模型被移动到：

```python
device
```

如果 CUDA 可用就是 GPU，否则使用 CPU。

### 22.1 MLP 如何看待最后一维

虽然通常把最后六个数称为“六个通道”，当前 MLP 的空间编码器会把输入：

```text
(B,T,F,A)
```

整理为：

```text
(B*T,1,F,A)
```

所以幅度和相位被拼接在最后一个空间维：

```text
A = 6
```

而 PyTorch Conv2d 的显式输入 channel 仍然是 `1`。

模型仍能在这一维上共同处理：

```text
前三个幅度位置
后三个相位位置
```

## 23. 第六步：训练

### 23.1 损失函数

```python
criterion = nn.CrossEntropyLoss()
```

用于九分类用户识别。

### 23.2 优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wd,
)
```

如果脚本没有显式覆盖，就从 Gait 参数配置中读取学习率和权重衰减。

### 23.3 学习率调度器

```python
ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=5,
)
```

监控：

```text
验证集 loss
```

如果验证 loss 连续若干轮没有改善，学习率乘以 `0.1`。

日志中显示：

```text
LR: 0.000000
```

时不一定严格等于零，也可能只是数值已经小于六位小数显示精度。

### 23.4 每个 epoch

训练阶段：

```text
model.train()
→ 取一个 train batch
→ 数据和标签移动到 GPU
→ 前向传播
→ 计算交叉熵
→ optimizer.zero_grad()
→ loss.backward()
→ optimizer.step()
→ 统计训练 loss 和 accuracy
```

验证阶段：

```text
model.eval()
→ torch.no_grad()
→ 遍历 val_loader
→ 只前向传播，不反向传播
→ 统计验证 loss 和 accuracy
```

日志里的：

```text
Duration
```

包括该轮训练集和验证集的时间，不包括最终测试集。

### 23.5 最佳 checkpoint

调度器监控的是验证 loss，但最佳 checkpoint 判断使用：

```python
if epoch_val_acc > best_val_acc:
```

也就是：

```text
验证准确率创新高
→ 保存 best_checkpoint.pth
```

checkpoint 包含：

```text
epoch
model_state_dict
optimizer_state_dict
scheduler_state_dict
best_val_acc
history
padding_length
```

### 23.6 loss 曲线

只有 `train_model()` 完成全部 epoch 并返回 `history` 后，才执行：

```python
save_loss_curve(...)
```

所以如果程序中途退出：

```text
可能已经有最佳 checkpoint
但不会生成完整 loss_curve.png
```

## 24. 第七步：测试

训练结束后：

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)
```

然后加载：

```python
model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

注意测试的不是最后一轮模型，而是验证准确率最高时保存的模型。

调用：

```python
_evaluate_model(
    model,
    test_loader,
    device,
)
```

评估过程：

```text
model.eval()
→ torch.no_grad()
→ 遍历测试集
→ 模型输出每个类别的分数
→ torch.max 取得预测类别
→ 汇总全部预测和标签
→ accuracy_score
```

最后打印：

```text
最佳验证准确率
测试集准确率
```

## 25. 默认 high_quality 的形状和类型时间线

```text
单帧 BfeeReader 输出
shape: (30,3)
dtype: complex64
原因: 同时保存 I/Q，即实部和虚部

按时间 stack
shape: (T,30,3)
dtype: complex64

Butterworth 降噪后
shape: (T,30,3)
dtype: complex

STC 相位校准后
shape: (T,30,3)
dtype: complex

特殊 z-score
shape: (T,30,6)
dtype: real float
内容: [signed_norm_amp, phase]

resize 后
shape: (1500,30,6)
dtype: real float

split 后
train/val/test 各自保持 (N,1500,30,6)

CSIDataset 后
shape: (N,1500,30,6)
dtype: torch.float32

DataLoader batch
shape: (B,1500,30,6)
dtype: torch.float32

模型输入
shape: (B,1500,30,6)
```

## 26. 如果换成其他 pipeline，会走哪个分支

### 26.1 `high_quality`、`robust` 等 z-score 预设

只要满足：

```text
dataset = gait/widar
normalize.method = z-score
```

就有：

```text
phase_zscore = True
```

流程：

```text
Processor 移除普通 normalize
→ 执行其他算法
→ 单独生成实数 [norm_amp, phase]
→ CSIDataset 直接保留
```

### 26.2 `fast` 的 min-max

`fast` 使用：

```python
"normalize": {
    "method": "min-max",
}
```

所以：

```text
pipeline_uses_zscore(...) = False
phase_zscore = False
```

Processor 不会移除 normalize，普通 pipeline 会执行 min-max。

由于 min-max 的归一化幅度位于 `[0,1]`，没有 z-score 负幅度问题。普通
`normalize_amplitude()` 会把相位重新乘回去，返回复数：

```text
norm_amp * exp(j*phase)
```

进入 `CSIDataset`：

```text
phase_dataset = True
phase_zscore = False
np.iscomplexobj(data_array) = True
```

因此进入：

```python
elif (
    phase_dataset
    and np.iscomplexobj(data_array)
):
```

执行：

```python
amplitude = np.abs(data_array)
phase = np.angle(data_array)
data_list = np.concatenate(
    [amplitude, phase],
    axis=-1,
)
```

最终同样得到：

```text
(1500,30,6) 实数幅度+相位
```

### 26.3 自定义 pipeline 不包含 normalize

例如：

```python
PIPELINE_STEPS = {
    "denoise": {
        "method": "savgol",
        ...
    }
}
```

此时：

```text
pipeline_steps_override is not None
→ True
```

自定义字典优先于 `PRESET_NAME`。

因为没有 z-score：

```text
phase_zscore = False
```

Processor 返回复数 Gait CSI，`CSIDataset` 自动执行：

```text
abs + angle + concatenate
```

### 26.4 BaseProcessor

设置：

```python
PRESET_NAME = "baseprocessor"
PIPELINE_STEPS = None
```

`resolve_pipeline_steps()` 返回：

```text
None
```

`process_data()` 选择：

```python
BaseProcessor()
```

BaseProcessor 对每个 Gait 样本执行：

```text
线性 phase_calibration
→ wavelet_denoise_csi
```

由于 Gait 需要相位，结果仍是复数。

构造 Dataset 时：

```text
pipeline_steps = None
phase_dataset = True
pipeline_uses_zscore(None) = False
phase_zscore = False
np.iscomplexobj(data_array) = True
```

所以 `CSIDataset` 自动拆成：

```text
[amplitude, phase]
```

因此当前全局策略下，Gait 使用 BaseProcessor 也会自动使用幅度和相位。

## 27. `CSIDataset` 四个分支总表

| 条件 | 实际行为 |
| --- | --- |
| Gait/Widar + z-score | 输入应已是实数 `[norm_amp, phase]`，直接保留 |
| Gait/Widar + 非 z-score + 复数 | 执行 `abs + angle + concatenate` |
| XRF55 + 实数 | 保留训练集归一化产生的正负号 |
| 其他情况 | 执行历史幅度路径 `np.abs(data_array)` |

对于本文默认 Gait high_quality，走的是第一行。

## 28. 关键 `if` 条件汇总

| 位置 | 条件 | 默认结果 | 原因 |
| --- | --- | --- | --- |
| `resolve_pipeline_steps` | override 非空 | False | `PIPELINE_STEPS=None` |
| `resolve_pipeline_steps` | preset 是 baseprocessor | False | 当前是 high_quality |
| `process_data` | pipeline_steps is None | False | 已解析为 high_quality 字典 |
| 单样本 Processor | 没有 frame | False | 正常文件有 CSI frame |
| 单样本 Processor | `whole_csi.ndim == 2` | False | 实际是 `(T,30,3)` |
| 单样本 Processor | 时间长度小于2 | False | 正常样本有大量 frame |
| `uses_phase_amplitude` | dataset 是 gait/widar | True | 当前是 gait |
| `pipeline_uses_zscore` | normalize.method 是 z-score | True | high_quality 使用 z-score |
| Processor | `phase_zscore` | True | 上述两个条件都成立 |
| Processor | XRF55 特殊条件 | False | 当前不是 XRF55 |
| 特殊 normalize | `return_phase_channels` | True | 需要保留负幅度和正确相位 |
| split | dataset 是 XRF55 | False | 当前是 gait |
| split | group 少于3个 | False | Gait 有多个条件组 |
| CSIDataset | `phase_zscore` | True | gait + high_quality |
| CSIDataset | 输入仍是复数 | False | Processor 已输出实数六维特征 |
| CSIDataset | 复数幅度相位拆分分支 | 不执行 | 第一个 `if` 已成立 |
| 训练 | CUDA 可用 | 通常 True | 使用 GPU 0 |
| 保存 checkpoint | val acc 创新高 | 动态判断 | 只有创新高才覆盖最佳模型 |

## 29. 最容易混淆的几个问题

### 29.1 `phase_zscore=True` 是否代表数据已经是实数

不一定。

刚计算出：

```python
phase_zscore = True
```

时，`whole_csi` 仍是原始复数 CSI。

它只代表：

```text
接下来需要执行特殊 z-score 处理。
```

调用：

```python
normalize_amplitude(
    ...,
    return_phase_channels=True,
)
```

以后，输出才变成实数。

### 29.2 为什么 Processor 和 Dataset 都有 `phase_zscore`

不是重复做同一件事。

```text
Processor 中的 phase_zscore：
决定如何生成数据。

Dataset 中的 phase_zscore：
决定如何接收和保护已经生成的数据。
```

### 29.3 为什么拼接相位后是实数

因为：

```python
norm_amp = 实数数组
phase = 实数数组
```

`np.concatenate()` 只是把两个实数数组放在一起：

```text
[-1.2, 0.8, ..., -2.4, 1.1, ...]
```

没有出现：

```python
1j
```

也没有调用：

```python
np.exp(1j * phase)
```

所以输出是实数。

### 29.4 `np.angle()` 输出相位，为什么它本身是实数

相位是一个角度数值，例如：

```text
0.7
-1.3
3.14
```

这些角度使用实数表示。

复数负责把“幅度和相位”编码到一个值里；拆开以后，幅度和相位本身都是实数。

### 29.5 当前 z-score 是训练集整体归一化吗

不是。

Gait 当前在：

```python
_process_single_csi_configurable()
```

中对每个 CSI 样本分别沿时间轴计算均值和标准差。

这发生在 train/val/test 划分之前，但每个样本只使用自己的时间数据，不会把其他
样本或测试集样本混进自己的统计量。

XRF55 才有划分后只使用训练集参数进行全局归一化的专用逻辑。

## 30. 最终结论

当前默认 `pipline_gait_steps.py` 的最关键流程是：

```text
Gait Reader 读取 complex64 CSI
→ high_quality 包含 z-score
→ uses_phase_amplitude("gait") = True
→ pipeline_uses_zscore(high_quality) = True
→ phase_zscore = True
→ 暂时移除普通 normalize
→ 执行 Butterworth 和 STC，数据仍是复数
→ 单独计算实数 norm_amp 和实数 phase
→ 拼接成实数 (T,30,6)
→ resize 成 (1500,30,6)
→ 按 track+receiver group 划分
→ CSIDataset 再次判断 phase_zscore=True
→ 确认输入不是复数
→ 直接保留 [signed_norm_amp, phase]
→ 转成 torch.float32
→ 模型训练、验证和测试
```

这套设计的核心目的不是简单地“增加相位”，而是同时保证：

```text
z-score 幅度的负号不丢失
相位不因为负幅度额外偏移 π
模型真正收到独立的幅度和相位特征
训练和测试使用完全相同的数据表示
```
