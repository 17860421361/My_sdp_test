# 修复前 Gait/Widar 为什么没有正确使用相位信息

## 1. 分析范围与历史版本

本文回答以下问题：

1. 原本的测试代码为什么没有使用到相位信息；
2. 原本的测试流程是什么；
3. 修复之前，测试代码和 WSDP 公共源码分别是什么行为；
4. 为什么不能只把 `USE_PHASE` 改成 `True`；
5. 从局部测试脚本修补到公共源码统一修复，分别解决了什么问题。

历史版本以 GitHub `main` 的提交记录为准：

- 初始版本：[e34573b Initial commit](https://github.com/17860421361/My_sdp_test/commit/e34573b)
- 全局修复前基线：[3f79cdb gait train inf](https://github.com/17860421361/My_sdp_test/commit/3f79cdb5c9ff60f9bcaf3c532e3843bd3d97ef5a)
- 幅度/相位全局修复：[9ae9ad5 fixed](https://github.com/17860421361/My_sdp_test/commit/9ae9ad5640eefcd79114df15cfa79b72ec953528)
- 合并到 `main`：[ce913bb Merge pull request #1](https://github.com/17860421361/My_sdp_test/commit/ce913bb)

这里所说的“修复前”，主要指 `9ae9ad5` 的父提交 `3f79cdb`。最早的
`test1_gait` 测试代码也包含在分析中，因为它能最直接地说明原始测试路径为什么
最终只向模型提供幅度。

## 2. 先说结论

原始测试过程中，相位校准算法通常确实执行了，但校准后的相位没有作为模型输入。

根因发生在构造 `CSIDataset` 时：

```python
CSIDataset(train_data, train_labels)
```

旧版 `CSIDataset` 的 `use_phase` 默认值是 `False`：

```python
def __init__(
    self,
    data_list,
    labels,
    use_phase=False,
    preserve_real_sign=False,
):
```

由于调用方没有传入 `use_phase=True`，最后进入：

```python
data_list = np.abs(data_array)
```

因此实际数据流是：

```text
原始复数 CSI
→ 降噪、相位校准、归一化等 pipeline 算法
→ 得到处理后的复数 CSI
→ CSIDataset 无条件走默认幅度分支
→ np.abs(...)
→ 相位被删除
→ 模型只收到幅度
```

所以需要区分两个说法：

- “相位校准算法没有执行”是不准确的；
- “相位校准执行了，但相位没有作为独立特征进入模型”才是准确描述。

修复前的 Gait 运行日志也给出了直接证据：

```text
样本形状: (1500, 30, 3)
是否使用相位信息：False
```

如果模型真正使用了三个幅度通道和三个相位通道，最后一维应当是 `6`，而不是
`3`。

## 3. 原本的批量测试逻辑

### 3.1 总体流程

最早的 Gait 批量测试文件是：

```text
test1_gait/full_test_presets_models_gait.py
```

后来的 step 版批量测试文件是：

```text
SDP/test_gait/full_test_presets_models_gait.py
SDP/test_wider/full_test_presets_models_widar.py
```

它们的总体实验结构相同：

```text
读取一次原始数据
→ 获取所有算法预设
→ 获取所有注册模型
→ 对每个预设执行一次数据处理
→ resize/padding 到固定时间长度
→ 映射标签和分组
→ 划分训练集、验证集、测试集
→ 构造 CSIDataset 和 DataLoader
→ 在同一个预设处理结果上依次训练所有模型
→ 使用验证集选择最佳 checkpoint
→ 使用最佳 checkpoint 在测试集评估
→ 保存日志、loss 曲线、checkpoint 和 summary CSV
```

一个预设只处理和划分一次数据，该预设下的所有模型共用相同的训练集、验证集和
测试集。

### 3.2 Gait 的标签和分组

Gait 使用：

```text
label = user_id
group = track_id * 100 + receiver_id
```

因此模型学习用户身份，数据划分则按“轨迹 + 接收端”组合进行
`GroupShuffleSplit`，避免同一个采集条件同时进入训练集和测试集。

### 3.3 Widar 的标签和分组

Widar 使用：

```text
label = gesture_type
group = position_id * 1000 + orientation_id * 100 + receiver_id
```

因此模型学习手势类别，数据按“位置 + 朝向 + 接收端”组合划分。

### 3.4 算法预设如何执行

批量脚本首先调用：

```python
pipeline_steps = apply_preset(preset_name)
```

例如：

```python
high_quality = {
    "denoise": {"method": "butterworth", ...},
    "calibrate": {"method": "stc"},
    "normalize": {"method": "z-score"},
}
```

随后把预设交给：

```python
processor = ConfigurableProcessor(pipeline_steps)
all_data, all_labels, all_groups = processor.process(
    csi_data_list,
    dataset=dataset_name,
)
```

`ConfigurableProcessor` 会对每个 CSI 样本调用 `execute_pipeline()`，按照规定的
顺序执行：

```text
denoise
→ outliers（如果存在）
→ calibrate
→ normalize
→ interpolate（如果存在）
→ 其他步骤
```

到这里为止，相位校准的确已经执行。

### 3.5 相位是在什么位置丢失的

最早的 Gait 批量代码构造 Dataset 的方式是：

```python
train_dataset = CSIDataset(train_data, train_labels)
valid_dataset = CSIDataset(valid_data, valid_labels)
test_dataset = CSIDataset(test_data, test_labels)
```

它没有传递：

```python
use_phase=True
```

所以旧版 `CSIDataset` 最终执行 `np.abs()`。这一步发生在数据已经完成 pipeline、
resize 和 split 之后，真正送入模型之前。

也就是说，问题并不在数据读取，也不在模型内部，而是在模型入口的数据表示转换。

## 4. 修复前公共源码逐文件说明

### 4.1 `wsdp/datasets/CSIDataset.py`

修复前文件：

```text
SDP/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/
src/wsdp/datasets/CSIDataset.py
```

核心逻辑是：

```python
if use_phase and np.iscomplexobj(data_array):
    amplitude = np.abs(data_array)
    phase = np.angle(data_array)
    data_list = np.concatenate([amplitude, phase], axis=-1)
elif preserve_real_sign:
    ...
else:
    data_list = np.abs(data_array)
```

这个实现本身具备手动启用相位的能力，但存在三个问题：

1. `use_phase=False` 是默认值；
2. `core.pipeline()` 和原始测试脚本没有为 Gait/Widar 传入 `True`；
3. 是否使用相位依赖每一个调用方手动设置，容易遗漏，训练和推理也可能不一致。

所以“源码里有幅度和相位拼接代码”不等于“实际实验使用了相位”。必须同时满足：

```text
use_phase=True
并且输入仍然是复数
```

原始测试不满足第一个条件。

### 4.2 `wsdp/processors/base_processor.py`

修复前的 `BaseProcessor` 会执行：

```python
whole_csi = phase_calibration(whole_csi, dataset=dataset)
cleaned_csi = wavelet_denoise_csi(whole_csi)
```

因此 BaseProcessor 不是没有做相位校准。问题是它返回复数 CSI 后，后面的
`CSIDataset` 又只取了绝对值。

相位校准通常可写成：

```text
H = A × exp(jφ)
校准后 H' = A × exp(jφ')
```

最后取绝对值：

```text
abs(H') = A
```

于是校准后的 `φ'` 被丢掉。

BaseProcessor 在相位校准后还会执行复数降噪，所以校准后的复数值有可能通过后续
降噪间接影响最终幅度；但模型仍然没有收到独立的相位通道。

### 4.3 `wsdp/processors/configurable_processor.py`

修复前的 `ConfigurableProcessor` 直接执行：

```python
cleaned_csi = execute_pipeline(
    whole_csi,
    effective_pipeline_steps,
    dataset=dataset,
)
```

当数据集是 XRF55 且使用 z-score/min-max 时，它会暂时移除 normalize，留到数据
划分后使用训练集参数完成归一化。

但是当数据集是 Gait/Widar 时，没有任何“自动生成幅度+相位通道”的公共策略。
它只返回 pipeline 处理后的复数 CSI，后续如何转换完全交给 `CSIDataset`。

### 4.4 `wsdp/algorithms/amplitude.py`

修复前 z-score 的核心实现是：

```python
amplitude = np.abs(csi)
mean = np.mean(amplitude, axis=0, keepdims=True)
std = np.std(amplitude, axis=0, keepdims=True)
norm_amp = (amplitude - mean) / std

if np.iscomplexobj(csi):
    phase = np.angle(csi)
    result = norm_amp * np.exp(1j * phase)
```

表面上看，这段代码“重新乘回了相位”，但 z-score 的 `norm_amp` 可能为负数。

假设：

```text
norm_amp = -3
原相位 = φ
```

复数结果是：

```text
-3 × exp(jφ)
```

它在复平面上等价于：

```text
3 × exp(j(φ + π))
```

如果后续再执行：

```python
np.abs(result)
np.angle(result)
```

得到的是：

```text
幅度 = 3
相位 = φ + π
```

而真正希望保留的是：

```text
带符号归一化幅度 = -3
相位 = φ
```

因此旧 z-score 复数表示存在两个问题：

- z-score 幅度的负号会被 `abs()` 删除；
- 负幅度会被编码成相位额外偏移 `π`。

这也是为什么不能简单地把 `use_phase` 改成 `True` 就结束修复。

对于 min-max，归一化幅度通常位于 `[0, 1]`，没有负号造成的 `π` 偏移问题；
但如果 `use_phase` 仍为默认 `False`，相位最终一样会被删除。

### 4.5 `wsdp/core.py`

修复前 `wsdp.core.pipeline()` 构造 Dataset 时只考虑了 XRF55 的符号保留：

```python
preserve_real_sign = (
    is_amplitude_primary_dataset(dataset_name)
    and ...
)

train_dataset = CSIDataset(
    train_data,
    train_labels,
    preserve_real_sign=preserve_real_sign,
)
```

它没有传入：

```python
use_phase=True
```

也没有把 `dataset_name` 和 `pipeline_steps` 交给 Dataset 判断。

所以即使不运行自定义测试文件，直接调用：

```python
wsdp.core.pipeline(...)
```

修复前的 Gait/Widar 仍然会走幅度输入路径。

旧 core 还使用：

```python
input_shape = train_data[0].shape
```

如果 Dataset 在内部把最后一维从 `A` 扩展为 `2A`，这个形状可能不再等于模型真正
收到的形状。对于依赖 `input_shape` 构造层的模型，会引入新的形状不一致问题。

### 4.6 `wsdp/inference.py`

修复前推理代码明确执行：

```python
data = np.abs(data)
tensor_data = torch.from_numpy(data).float()
```

因此即使某个本地训练脚本手动加入了相位，公共推理接口仍会无条件删除相位。

这会形成：

```text
本地训练：可能使用幅度+相位
公共推理：只使用幅度
```

训练和推理的数据表示不一致。

### 4.7 `wsdp/dataset_policy.py`

修复前只有：

```python
AMPLITUDE_PRIMARY_DATASETS = {"xrf55"}
```

它只描述 XRF55 的幅度数据策略，没有：

```python
PHASE_AMPLITUDE_DATASETS = {"widar", "gait"}
```

因此公共源码不知道哪些数据集应该自动使用相位，判断只能散落在各测试脚本中。

## 5. 相位校准算法到底有没有起作用

预设中的相位校准方法包括：

```text
linear
polynomial
stc
robust
```

修复前它们通常会被 `execute_pipeline()` 正常调用，并生成校准后的复数 CSI。

但是这些算法的主要输出变化位于复数相位。最终 `CSIDataset` 执行 `np.abs()` 后，
模型看不到校准相位。

因此更准确的判断是：

```text
相位校准算法执行了
≠
模型使用了校准后的相位
```

对于只改变相位并保持幅度的校准：

```text
校准前：A × exp(jφ)
校准后：A × exp(jφ')
取幅度后：A
```

从模型最终输入看，校准前后相同。

需要保留一个技术上的细节：如果相位校准之后还有复数降噪、复数插值等会混合
多个复数值的算法，相位可能间接影响后续计算得到的幅度。但这仍不代表模型得到
了独立相位通道。可以确定的是，最终相位值本身没有进入模型。

同时，一个预设可能还包含降噪、归一化、插值等算法，所以旧实验中不同预设得到
不同准确率，不能说明相位校准已经作为模型特征生效。差异可能来自预设里的其他
算法。

## 6. 局部脚本修补阶段是什么状态

在全局源码修复之前，Gait/Widar 的 step 脚本已经加入了局部兼容逻辑：

```text
SDP/test_gait/pipline_gait_steps.py
SDP/test_wider/pipline_widar_steps.py
```

### 6.1 默认仍然没有使用相位

修复前配置是：

```python
USE_PHASE = False
```

批量脚本复用 step 脚本的 `build_loaders()`，所以默认批量实验仍然只使用幅度。

这也是旧结果目录使用以下名称的原因：

```text
Gait:  user_id_v2
Widar: source_gesture_condition
```

它们属于旧的 amplitude-only 实验。

### 6.2 非 z-score 的局部处理

如果手动把：

```python
USE_PHASE = True
```

并且 pipeline 不含 z-score，旧 step 脚本会调用：

```python
CSIDataset(
    data,
    labels,
    use_phase=True,
)
```

此时可以得到：

```text
[abs(CSI), angle(CSI)]
```

这个分支能够让非 z-score 的 Gait/Widar 使用幅度和相位。

### 6.3 z-score 的局部特殊处理

当同时满足：

```text
USE_PHASE=True
pipeline 包含 z-score
```

旧 step 脚本会：

1. 临时从 pipeline 中移除 normalize；
2. 先执行降噪、相位校准、插值等其他步骤；
3. 单独计算带正负号的 z-score 幅度；
4. 直接拼接 `[normalized_amplitude, phase]`；
5. 让 `CSIDataset` 保留这个已经准备好的实数数组。

核心形式是：

```python
normalized_amplitude = (amplitude - mean) / std
data = np.concatenate(
    [normalized_amplitude, phase],
    axis=-1,
)
```

这能避开“负幅度被编码成相位偏移 `π`”的问题。

### 6.4 为什么局部方案还不够

局部方案可以让指定 step 脚本工作，但它不是完整的公共解决方案：

- Gait 和 Widar 分别复制一套判断；
- 单组脚本、批量脚本、core、benchmark、超参数搜索都可能遗漏；
- 公共推理仍然无条件执行 `np.abs()`；
- `USE_PHASE=False` 很容易被忘记修改；
- Dataset 本身不知道数据集策略；
- 训练与推理可能使用不同表示；
- 输入形状和旧缓存也可能沿用幅度-only 版本。

所以后续才需要修改公共源码，而不只是继续维护每个测试文件里的
`manual_phase_zscore`。

## 7. 全局修复做了什么

提交 `9ae9ad5` 把分散在测试脚本中的判断统一移入公共源码。

### 7.1 数据集策略

在 `dataset_policy.py` 中新增：

```python
PHASE_AMPLITUDE_DATASETS = {"widar", "gait"}
```

并通过：

```python
uses_phase_amplitude(dataset_name)
pipeline_uses_zscore(pipeline_steps)
```

统一判断。

### 7.2 z-score 输出

`normalize_amplitude()` 增加：

```python
return_phase_channels=True
```

启用后直接返回实数形式：

```text
[带符号的 norm_amp, 正确的 phase]
```

不再把负幅度重新编码进复数相位。

### 7.3 Processor

当检测到：

```text
数据集是 Gait/Widar
并且 pipeline 使用 z-score
```

`ConfigurableProcessor` 会暂时移除普通 normalize，执行其他算法后，再生成实数
形式的幅度相位通道。

### 7.4 Dataset

新的接口是：

```python
CSIDataset(
    data,
    labels,
    dataset_name=DATASET_NAME,
    pipeline_steps=pipeline_steps,
)
```

不再依赖 `use_phase`：

- Gait/Widar + 非 z-score：自动执行 `abs + angle + concatenate`；
- Gait/Widar + z-score：直接保留 Processor 准备好的实数幅度相位通道；
- XRF55/ElderAL：保持幅度输入策略。

### 7.5 core 与推理

`core.pipeline()` 和 `inference.py` 都改为走同一个 Dataset 策略，并从 Dataset
读取模型真正收到的输入形状：

```python
input_shape = tuple(train_dataset.data_list.shape[1:])
```

因此训练和推理不再分别维护两套幅度/相位规则。

### 7.6 实验标识

修复后的批量实验使用新的 `RUN_NAME`：

```text
Gait:  user_id_v3_auto_amp_phase
Widar: source_gesture_condition_auto_amp_phase
```

这样旧的 amplitude-only summary 不会让断点续跑逻辑误判新实验已经完成，也不会
覆盖旧 checkpoint。

## 8. 修复前后完整数据流对照

### 8.1 修复前

```text
复数原始 CSI
→ Processor
→ denoise
→ calibrate（相位校准确实执行）
→ normalize
→ resize/padding
→ group split
→ CSIDataset(use_phase=False)
→ np.abs
→ (T, F, A)
→ 模型
```

模型输入：

```text
只有幅度
```

### 8.2 修复后：非 z-score

```text
复数原始 CSI
→ Processor
→ pipeline
→ resize/padding
→ group split
→ CSIDataset 根据 dataset_name 判断 Gait/Widar
→ concatenate([abs(CSI), angle(CSI)])
→ (T, F, 2A)
→ 模型
```

### 8.3 修复后：z-score

```text
复数原始 CSI
→ ConfigurableProcessor 识别 Gait/Widar + z-score
→ 暂时移除普通 normalize
→ 执行其他 pipeline 算法
→ 计算带正负号的 z-score 幅度
→ 拼接 [normalized_amplitude, phase]
→ 得到实数 (T, F, 2A)
→ resize/padding
→ group split
→ CSIDataset 直接保留
→ 模型
```

## 9. 如何理解旧实验结果

旧 Gait/Widar 结果应标记为：

```text
amplitude-only
```

即使预设名称里包含：

```text
calibrate=linear/polynomial/stc/robust
```

也不能把结果解释成“模型使用了相位”。最多只能说明该完整预处理组合在最终幅度
输入上的实验结果。

旧模型输入通常是：

```text
(T, F, A)
```

新模型输入是：

```text
(T, F, 2A)
```

例如 Gait：

```text
旧输入：(1500, 30, 3)
新输入：(1500, 30, 6)
```

因此旧 checkpoint 与新幅度相位模型并不是同一种输入定义。对于输入层依赖
`input_shape` 的模型，旧 checkpoint 不能直接加载到新模型中继续训练或推理。

## 10. 最终回答

原本没有正确使用相位，不是因为原始数据没有相位，也不是因为预设中的相位校准
没有运行，而是因为：

1. 测试代码创建 `CSIDataset` 时没有传入 `use_phase=True`；
2. `use_phase` 默认是 `False`；
3. `CSIDataset` 在模型入口执行 `np.abs()`，删除了相位；
4. 公共 `core.pipeline()` 同样没有 Gait/Widar 的自动相位策略；
5. 公共推理接口也无条件执行 `np.abs()`；
6. 简单打开 `use_phase` 后，z-score 负幅度还会造成负号丢失和相位偏移 `π`。

所以完整修复必须同时覆盖：

```text
数据集策略
→ Processor 的 z-score 表示
→ CSIDataset 的模型输入
→ core 的输入形状和缓存
→ inference 的一致表示
→ Gait/Widar 实验入口
```

这也是为什么最终选择修改公共源码，而不是只在某一个测试脚本中增加一个
`USE_PHASE=True`。
