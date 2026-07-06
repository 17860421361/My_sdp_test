# PR Description：基于 dataset_name 自动使用幅度/相位输入

## 修改目的

本 PR 主要修复 Gait / Widar 数据集没有稳定、统一使用相位信息的问题，并把原本分散在各个测试脚本里的 `USE_PHASE`、`manual_phase_zscore`、`preserve_real_sign` 等逻辑统一收到 WSDP 公共源码中。

修改后不再由每个实验脚本手动决定是否使用相位，而是由 `dataset_name` 自动决定输入形式：

| 数据集 | 模型输入 |
| --- | --- |
| `gait` | 幅度 + 相位 |
| `widar` | 幅度 + 相位 |
| `xrf55` | 幅度 |
| `elderAL` | 幅度 |

这样可以避免不同测试脚本之间处理逻辑不一致，也能保证 Gait / Widar 的模型真正收到相位通道。

## 核心问题

原逻辑里，即使数据本身是复数 CSI，很多路径最终仍然会在 `CSIDataset` 中执行：

```python
data_list = np.abs(data_array)
```

这会只保留幅度，丢掉相位。

另外，对于 Gait / Widar 这类需要使用相位的数据，如果 pipeline 中包含 `z-score`，原来的 `normalize_amplitude()` 会把归一化后的幅度重新构造成复数：

```python
norm_amp * np.exp(1j * phase)
```

但是 `z-score` 后的 `norm_amp` 可能是负数。比如：

```text
-3 * exp(1j * phase)
```

在复数表达中等价于：

```text
3 * exp(1j * (phase + π))
```

后续如果再执行：

```python
amplitude = np.abs(data_array)
phase = np.angle(data_array)
```

就会变成：

```text
幅度: 3
相位: phase + π
```

这会导致两个问题：

1. z-score 幅度的正负号丢失；
2. 原始相位被偏移 π。

因此 Gait / Widar + z-score 需要单独处理，不能先构造成复数再交给 `CSIDataset` 重新 `abs + angle`。

## 主要修改文件

### 1. `wsdp/dataset_policy.py`

新增数据集输入策略判断：

```python
PHASE_AMPLITUDE_DATASETS = {"widar", "gait"}
```

新增公共函数：

```python
uses_phase_amplitude(dataset_name)
pipeline_uses_zscore(pipeline_steps)
```

作用：

- `uses_phase_amplitude()` 用来判断当前数据集是否应该自动使用“幅度 + 相位”；
- `pipeline_uses_zscore()` 用来判断当前 pipeline 是否包含 z-score 归一化。

这样相位相关判断不再散落在各个测试脚本里。

### 2. `wsdp/algorithms/amplitude.py`

给 `normalize_amplitude()` 增加参数：

```python
return_phase_channels=False
```

默认值为 `False`，保持原有行为不变。

当设置为 `True` 时，不再返回复数，而是直接返回实数特征：

```python
np.concatenate([norm_amp, phase], axis=-1)
```

也就是：

```text
[带正负号的 z-score 幅度, 原始/校准后的相位]
```

这样可以同时保留：

- z-score 后幅度的正负号；
- 正确的相位信息。

### 3. `wsdp/processors/configurable_processor.py`

新增 Gait / Widar + z-score 的自动判断：

```python
phase_zscore = (
    uses_phase_amplitude(dataset)
    and pipeline_uses_zscore(pipeline_steps)
)
```

当 `phase_zscore=True` 时：

1. 先从 pipeline 中临时移除 `normalize`；
2. 正常执行其他算法，例如 denoise、calibrate、interpolate；
3. 最后单独调用：

```python
normalize_amplitude(
    cleaned_csi,
    method="z-score",
    return_phase_channels=True,
)
```

最终返回的是实数形式的：

```text
[normalized_amplitude, phase]
```

而不是复数 CSI。

这样可以避免 z-score 负数幅度被复数表示转换成相位偏移。

### 4. `wsdp/datasets/CSIDataset.py`

删除原来的局部开关：

```python
use_phase
preserve_real_sign
```

新的公共接口为：

```python
CSIDataset(
    data_list,
    labels,
    dataset_name="",
    pipeline_steps=None,
)
```

内部根据 `dataset_name` 和 `pipeline_steps` 自动选择处理逻辑。

#### Gait / Widar + z-score

此时 `ConfigurableProcessor` 已经提前生成了实数特征：

```text
[normalized_amplitude, phase]
```

所以 `CSIDataset` 直接保留：

```python
data_list = data_array
```

如果这条路径下传进来的仍然是复数，会直接报错，防止错误被静默掩盖。

#### Gait / Widar + 非 z-score

如果输入仍然是复数 CSI，则在 `CSIDataset` 中转换为：

```python
amplitude = np.abs(data_array)
phase = np.angle(data_array)
data_list = np.concatenate([amplitude, phase], axis=-1)
```

最终模型输入最后一维从 `A` 变为 `2A`。

#### XRF55

XRF55 不增加相位通道。对于训练集归一化后产生的实数数据，直接保留正负号，不再默认取绝对值。

#### ElderAL

ElderAL 仍然保持幅度输入，不增加相位通道。

### 5. `wsdp/core.py`

`pipeline()` 创建 Dataset 时，现在会传入：

```python
dataset_name=dataset_name
pipeline_steps=resolved_pipeline_steps
```

这样 `CSIDataset` 能知道当前数据集和 pipeline 配置，从而自动决定输入形式。

同时模型输入形状改为从真正进入模型的 Dataset Tensor 中读取：

```python
input_shape = tuple(train_dataset.data_list.shape[1:])
```

不再使用：

```python
train_data[0].shape
```

原因是对于 Gait / Widar，某些路径下相位是在 `CSIDataset` 中加入的。比如原始复数 CSI 是：

```text
(T, F, A)
```

进入 `CSIDataset` 后会变成：

```text
(T, F, 2A)
```

所以模型应该使用 Dataset 处理后的真实输入形状。

另外，Gait / Widar 的缓存 key 中加入了：

```text
automatic_amplitude_phase_v1
```

用于避免新代码错误复用旧的 amplitude-only 缓存。

## 修改后的整体数据流

### Gait / Widar，pipeline 不包含 z-score

```text
复数 CSI
→ Processor 执行 pipeline
→ resize / split
→ CSIDataset 判断为 Gait/Widar
→ abs(CSI) 得到幅度
→ angle(CSI) 得到相位
→ concatenate([amplitude, phase], axis=-1)
→ 模型
```

输入形状变化：

```text
(T, F, A) → (T, F, 2A)
```

### Gait / Widar，pipeline 包含 z-score

```text
复数 CSI
→ 判断为 Gait/Widar + z-score
→ Processor 临时移除 normalize
→ 执行 denoise / calibrate / interpolate 等其他算法
→ 单独计算 z-score 幅度
→ 拼接 [normalized_amplitude, phase]
→ resize / split
→ CSIDataset 直接保留实数特征
→ 模型
```

最终模型输入为：

```text
[带正负号的 z-score 幅度, 正确相位]
```

## 对其他数据集的影响

### XRF55

XRF55 不使用相位，仍然是幅度输入。

同时保留原有的训练集归一化逻辑：使用训练集统计量归一化训练集，再用同一组参数处理验证集和测试集。

对于归一化后可能出现的实数正负号，`CSIDataset` 会保留，不会默认取 `abs()`。

### ElderAL

ElderAL 不使用相位，仍然保持幅度输入路径，行为基本不变。

## 实验脚本同步修改

四个 step 测试脚本同步改为传递：

```python
dataset_name=DATASET_NAME
pipeline_steps=pipeline_steps
```

包括：

```text
SDP/test_gait/pipline_gait_steps.py
SDP/test_wider/pipline_widar_steps.py
SDP/test_elderAL/pipline_elderAL_steps.py
SDP/test_xrf55/pipline_xrf55_steps.py
```

Gait / Widar 脚本中删除了本地的：

```text
USE_PHASE
manual_phase_zscore
preserve_real_sign
```

相位逻辑统一交给公共源码处理。

批量测试脚本也统一改为从 DataLoader 中读取真实输入形状：

```python
input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
```

避免 Gait / Widar 加入相位后模型仍然用旧形状初始化。

## 总结

这个 PR 的核心变化是：

1. Gait / Widar 自动使用“幅度 + 相位”输入；
2. XRF55 / ElderAL 保持幅度输入；
3. 修复 Gait / Widar + z-score 时负幅度导致相位偏移的问题；
4. 删除测试脚本里的局部相位开关，把逻辑统一放到 WSDP 公共源码；
5. 模型输入形状统一从 Dataset 实际输出读取，避免通道数变化导致模型初始化错误；
6. 增加缓存版本标记，避免新旧输入表示混用。

## Checklist

- [x] Gait / Widar 自动使用 amplitude + phase
- [x] 修复 z-score 负幅度导致的 phase shift 问题
- [x] XRF55 / ElderAL 保持 amplitude-only
- [x] `CSIDataset` 接口统一为 `dataset_name + pipeline_steps`
- [x] 模型输入形状从 Dataset 实际输出读取
- [x] 更新 Gait / Widar 缓存 key，避免复用旧缓存

## 26 种算法可用性验证

本轮额外对注册表中的 26 种算法做了单独可用性检查。判定标准是算法能否走完整个训练数据通路：

```text
读取真实 CSI
→ ConfigurableProcessor 的单样本处理逻辑
→ pipeline
→ resize_csi_to_fixed_length
→ CSIDataset
→ 模型训练
```

这里的“可用”只表示当前流程可以继续执行，不代表准确率一定更高，也不代表该算法在对应数据集上一定有充分物理意义。部分算法虽然能运行，但可能因为数据集本身缺少相位、采样率不匹配、子载波坐标没有显式传入等原因，实际作用有限。

### 去噪算法

| 去噪算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| wavelet | 可用 | 可用 | 可用 | 可用 | ElderAL 是实数幅度，当前 wavelet 可能产生复数形式，可能制造没有物理意义的“相位” |
| butterworth | 可用 | 可用 | 可用 | 可用 | 无 |
| savgol | 可用 | 可用 | 可用 | 可用 | 短序列时会自动缩小窗口，过短时可能近似不处理 |
| bandpass | 可用 | 可用 | 可用 | 可用 | 默认 `fs=1000`，Widar/Gait 基本匹配；XRF55 真实采样率不明确；ElderAL 采样不规则且明显不是 1000 Hz，保留和滤除的实际运动频率可能错误，准确率可能下降，也可能碰巧提高 |
| hampel | 可用 | 可用 | 可用 | 可用 | 能运行，但逐时间点、子载波计算，速度会比较慢 |

ElderAL 在这几个算法上会大量触发短序列保护，有的会直接返回原数据，有的会自动缩小窗口等。整体上能用，但作用不大，或者基本上没什么用。Bandpass 对 ElderAL 的实际作用最小，约 84% 样本因为不足 28 帧直接返回原数据。

### 异常值处理算法

| 异常值处理算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| iqr | 可用 | 可用 | 可用 | 可用 | 保持输入形状，只裁剪异常幅度 |
| z-score outlier | 可用 | 可用 | 可用 | 可用 | 属于 `outliers` 类，不是 `normalize` 中的 `z-score` |

### 相位校准算法

| 相位校准算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| linear | 可用 | 可用 | 可用 | 可用 | XRF55/ElderAL 没有真实相位，所以相位校准没用，通常原样返回；Widar/Gait 如果没有收到正确的数据集子载波坐标，会使用通用均匀坐标，效果可能差一点 |
| polynomial | 可用 | 可用 | 可用 | 可用 | 原因同上 |
| stc | 可用 | 可用 | 可用 | 可用 | 原因同上 |
| robust | 可用 | 可用 | 可用 | 可用 | `robust` 不依赖子载波坐标，但对 XRF55/ElderAL 这种实数幅度数据没有实际相位校准意义 |

XRF55 和 ElderAL 的本地数据是实数幅度，没有真实复数相位。因此相位校准算法虽然能跑通，但多数情况下只是 no-op，不能解释成相位校准真的有效。Widar/Gait 有复数 CSI，相位校准有意义，但建议后续测试时显式传入 IWL5300 子载波坐标，避免退回通用均匀坐标。

### 归一化算法

| 归一化算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| z-score | 可用 | 可用 | 可用 | 可用 | 有一些已经做过的特殊处理，比如 Widar/Gait 会使用当前源码中的幅度+相位通道逻辑 |
| min-max | 可用 | 可用 | 可用 | 可用 | 无 |
| agc | 不可用 | 不可用 | 不可用 | 不可用 | `wsdp/processors/configurable_processor.py` 中只保留了 `frame.csi_array`，没有保留 `frame.agc`，直接报错 |

AGC 的算法函数需要逐帧 `agc_values`，但当前 Processor 进入 pipeline 时只传 CSI 数组，不再携带每帧的 AGC 元数据。因此即使原始帧里有 `frame.agc`，普通 `pipeline_steps` 也拿不到它。在不修改 Processor 或不另写专门 adapter 的情况下，AGC 应记录为 `preprocess_error`，不进入后续大规模训练组合。

### 插值算法

| 插值算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| linear | 可用 | 可用 | 可用 | 可用 | Widar/Gait 如果没有收到正确的 IWL5300 子载波坐标，会使用默认均匀坐标 |
| cubic | 可用 | 可用 | 可用 | 可用 | 原因同上 |
| nearest | 可用 | 可用 | 可用 | 可用 | 原因同上 |
| decimate | 可用 | 可用 | 可用 | 可用 | 只能降采样，必须显式提供 `target_K`，且 `target_K <` 当前子载波数 `F` |

XRF55、Widar、Gait 的原始子载波数都是 `F=30`，因此使用 `linear`、`cubic` 或 `nearest` 且设置 `target_K=30` 时，会直接返回输入副本，完全不执行插值。ElderAL 原始为 512 子载波，所以 `target_K=30` 会执行插值；只有 `target_K=512` 才会直接返回。

建议后续把插值实验明确拆开：

| 数据集 | 原始 F | 上采样测试 | 降采样测试 |
| --- | ---: | ---: | ---: |
| XRF55 | 30 | 64 | 15 |
| Widar | 30 | 64 | 15 |
| Gait | 30 | 64 | 15 |
| ElderAL | 512 | 不建议上采样 | 64 或 30 |

### 特征提取算法

| 特征提取算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| doppler | 不可用 | 不可用 | 不可用 | 不可用 | 输出四维，当前三维 resize 直接报错 |
| entropy | 不可用 | 不可用 | 不可用 | 不可用 | 输出二维，当前 resize 直接报错 |
| ratio | 可用 | 可用 | 可用 | 可用 | 四个本地数据集都有至少两根天线，能运行；实数数据上只剩幅度比值 |
| decomposition | 不可用 | 可用 | 可用 | 不可用 | 实数输入时当前 CP 路径返回 `dict`，resize 需要 `ndarray` |
| conjugate_multiply | 不可用 | 可用 | 可用 | 不可用 | 要求复数 CSI，XRF55/ElderAL 是实数 |
| pca_fusion | 可用 | 条件 | 条件 | 可用 | Widar/Gait 与 `z-score` 组合会报错，与 `min-max` 或不归一化组合可以运行 |

Doppler 和 entropy 的函数本身可以计算，但输出表示已经不是 `(T, F, A)` 三维 CSI，当前分类训练管线无法直接接收。它们更适合单独做特征表示实验，或者后续增加专门的 adapter 后再接分类模型。

`pca_fusion` 单独运行可以生成三维数组，但 Widar/Gait 使用 `z-score` 时，当前 Processor 会把归一化推迟到 pipeline 结束，并要求输出仍然是复数 CSI，以生成 `[带符号归一化幅度, 相位]`。PCA 输出是实数，因此以下组合会报错：

```text
Widar + z-score + pca_fusion
Gait  + z-score + pca_fusion
```

以下组合可以运行：

```text
Widar/Gait + min-max + pca_fusion
Widar/Gait + 不归一化 + pca_fusion
```

### 检测算法

| 检测算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
| --- | --- | --- | --- | --- | --- |
| activity | 不可用 | 不可用 | 不可用 | 不可用 | 输出 `(T,)` 布尔序列，不是三维 CSI |
| change_point | 不可用 | 不可用 | 不可用 | 不可用 | 输出长度不固定的变点索引 |

检测算法回答的是“什么时候出现活动”或“什么时候发生状态变化”，而当前分类模型回答的是“这个完整样本属于哪个动作或用户”。任务定义不同，所以当前不应把它们直接纳入分类训练组合。

### 当前结论汇总

当前四个数据集上明确不能直接进入现有分类训练流程的项目是：

```text
normalize: agc
extract_features: doppler
extract_features: entropy
detect: activity
detect: change_point
```

只在 XRF55、ElderAL 上不能直接使用的项目是：

```text
extract_features: decomposition
extract_features: conjugate_multiply
```

需要特别标记条件或解释的组合是：

```text
Widar + z-score + pca_fusion
Gait  + z-score + pca_fusion
decimate + target_K >= 原始 F
XRF55/ElderAL + 任意相位校准
Widar/Gait + 未显式提供真实子载波坐标的相位校准或插值
采样率不明确的数据集 + bandpass
短序列 + 高阶滤波或大窗口平滑
```

因此，本轮测试不应该默认“注册表里的 26 个算法都能训练”。更准确的结论是：多数保持三维 CSI 的预处理算法可以直接进入当前流程；AGC、检测类、Doppler、entropy，以及部分依赖复数输入或输出表示变化的特征算法，需要单独记录失败原因或增加专门 adapter 后再纳入训练。
