# WSDP 算法可用性与组合测试建议（V2）

## 1. 先说结论

在不修改 WSDP 源码的情况下，可以完成当前最重要的目标：

> 找出哪些算法和组合能够走完整个训练流程，哪些会报错，以及哪些虽然能运行但基本没有作用。

但“不修改源码”不等于能让所有算法都正常进入现有分类训练流程：

- 有些问题可以通过测试配置绕开，例如显式传入子载波索引、选择正确的 `target_K`。
- 有些算法本身能计算，但输出形状不符合当前分类模型，例如 Doppler、entropy 和 detect。这些算法在现有训练流程中会报错。
- AGC 所需的逐帧元数据已经在 Processor 中丢失，单靠普通 `pipeline_steps` 无法补回来。

因此，本轮测试不应该事先假定“注册表中的算法都能训练”，而应当保留失败记录。失败本身就是测试结果。

## 2. 本文的判定标准

为了避免把“效果不好”和“不能运行”混在一起，本文使用四种状态。

| 状态 | 含义 |
|---|---|
| 可用 | 在当前源码中能够完成预处理、resize、构造 `CSIDataset`，可以继续训练 |
| 条件可用 | 参数、数据类型或组合满足条件时可以运行，否则会报错 |
| 能运行但可能无效 | 不报错，但可能直接返回原数据、使用了不准确的坐标，或算法的物理前提不成立 |
| 当前不可用于分类训练 | 在现有 Processor、resize 或 `CSIDataset` 流程中必然报错 |

这里的“可用”只表示能够执行，不代表准确率一定高，也不代表该算法适合这个数据集。

例如：

- XRF55 上的相位校准不报错，因此属于“能运行”。
- 但 XRF55 本地数据是实数幅度，没有真实相位，因此相位校准基本不起作用。
- 这种情况不能记为“算法不可用”，应记为“运行成功，但无效或意义有限”。

## 3. 验证方式

本文使用当前工作区里的最终版源码，并分别读取 XRF55、Widar、Gait、ElderAL 的真实样本，执行以下流程：

```text
读取真实 CSI
→ ConfigurableProcessor 的单样本处理逻辑
→ execute_pipeline
→ resize_csi_to_fixed_length
→ CSIDataset
```

因此，下表中的“报错”不是只看函数定义推测，而是按当前训练数据通路实际验证得到的。

这只是方法级冒烟测试。完整训练时还可能出现模型显存不足、模型不接受某种输入形状等问题。此类错误应记为“模型兼容问题”，不能直接认定为算法不可用。

## 4. 26 个注册算法的当前可用性

符号说明：

- `可用`：当前分类训练流程可继续运行。
- `有限`：能够运行，但可能无效、坐标不准确或物理意义有限。
- `条件`：满足表中条件才可用。
- `不可用`：当前分类训练流程会直接报错。

### 4.1 去噪算法

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| wavelet | 可用 | 可用 | 可用 | 有限 | ElderAL 是实数幅度；当前 wavelet 会产生复数形式，可能制造没有物理意义的“相位” |
| butterworth | 可用 | 可用 | 可用 | 可用 | 输出仍是三维 CSI |
| savgol | 可用 | 可用 | 可用 | 可用 | 短序列时会自动缩小窗口，过短时可能近似不处理 |
| bandpass | 有限 | 可用 | 可用 | 有限 | 默认 `fs=1000`；Widar/Gait 基本匹配，XRF55 真实采样率不明确，ElderAL 采样不规则且明显不是 1000 Hz |
| hampel | 可用 | 可用 | 可用 | 可用 | 能运行，但逐时间点、子载波、天线计算，中大型数据集上会比较慢 |

去噪类五种方法都不会因为输出形状而破坏当前训练流程。

### 4.2 异常值处理

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| iqr | 可用 | 可用 | 可用 | 可用 | 保持输入形状，只裁剪异常幅度 |
| z-score outlier | 可用 | 可用 | 可用 | 可用 | 这里属于 `outliers` 类，不是 normalize 中的 z-score |

这两种方法都可以进入组合测试。

### 4.3 相位校准

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| linear | 有限 | 有限 | 有限 | 有限 | XRF55/ElderAL 没有真实相位；Widar/Gait 当前没有自动收到正确的数据集子载波坐标 |
| polynomial | 有限 | 有限 | 有限 | 有限 | 原因同上；当前 registry 也不会自动传 dataset |
| stc | 有限 | 有限 | 有限 | 有限 | 原因同上 |
| robust | 有限 | 可用 | 可用 | 有限 | robust 不依赖子载波坐标；但对实数幅度数据没有真实相位可校准 |

四种相位算法都会运行，不属于“直接报错”的算法。

但是需要按数据集解释结果：

#### XRF55

本地 `.npy` 是非负实数幅度。相位校准函数检测到实数输入后会返回原数据，所以：

```text
XRF55 + linear/polynomial/STC/robust
```

大多是可运行的 no-op，即运行成功但基本没有处理效果。

#### ElderAL

ElderAL 也是实数幅度，相位校准同样没有真实物理相位可用。

如果前面先用了 wavelet，当前实现可能把实数包装成复数，后面的 STC 或 robust 可能产生数值变化。但这种变化来自算法内部的数据类型转换，不代表 ElderAL 突然获得了真实相位。此类组合即使不报错，也不建议把结果解释成“相位校准有效”。

#### Widar/Gait

它们具有真实复数 CSI，相位校准有意义。但当前 registry 只在 `dataset == "xrf55"` 时自动向部分算法传 dataset，因此 linear、polynomial、STC 在 Widar/Gait 上会退回通用的均匀子载波坐标。

算法仍然能运行，只是校准坐标不够准确，可能导致效果变差。

### 4.4 幅度归一化

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| z-score | 可用 | 可用 | 可用 | 可用 | Widar/Gait 会使用当前源码中的幅度+相位通道逻辑 |
| min-max | 可用 | 可用 | 可用 | 可用 | 能完成训练数据构造 |
| agc | 不可用 | 不可用 | 不可用 | 不可用 | 当前 Processor 没有向函数提供逐帧 `agc_values`，直接报 `TypeError` |

AGC 的函数定义需要：

```python
agc_compensate(csi, agc_values)
```

但是 Processor 只保留：

```python
frame.csi_array
```

没有把每一帧的：

```python
frame.agc
```

传进 pipeline。因此即使 Widar/Gait 的原始 `BfeeFrame` 含有 AGC，进入算法时也已经拿不到了。

在不修改源码或不另写自定义 Processor 的情况下，AGC 应：

1. 单独做一次预处理失败测试；
2. 记录为 `preprocess_error`；
3. 不进入后续大规模训练组合。

这不妨碍完成“找出哪些算法不能用”的实验目标。

### 4.5 插值与降采样

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| linear | 可用 | 有限 | 有限 | 可用 | Widar/Gait 当前不会自动取得正确的 IWL5300 子载波坐标 |
| cubic | 可用 | 有限 | 有限 | 可用 | 原因同上 |
| nearest | 可用 | 有限 | 有限 | 可用 | 原因同上 |
| decimate | 条件 | 条件 | 条件 | 条件 | 必须显式提供 `target_K`，且 `target_K < F` |

#### `target_K == F` 为什么没有效果

`interpolate_grid()` 中有明确逻辑：

```python
if F == target_K:
    return csi.copy()
```

所以 XRF55、Widar、Gait 原始都是 30 子载波时：

```python
{"interpolate": {"method": "cubic", "target_K": 30}}
```

不会报错，但输出与输入完全相同。这属于“能用但没有处理效果”，不是不可用。

建议测试：

```text
linear64
cubic64
nearest64
```

而不是把 `cubic30` 当成一项有效插值实验。

#### decimate 的参数限制

decimate 只能降采样：

```text
target_K < 当前子载波数 F
```

合理示例：

```python
# XRF55/Widar/Gait: F=30
{"interpolate": {"method": "decimate", "target_K": 15}}

# ElderAL: F=512
{"interpolate": {"method": "decimate", "target_K": 64}}
```

以下配置会按设计报错：

```text
F=30, target_K=30
F=30, target_K=64
```

这属于参数不满足算法约束，不应认定为 decimate 本身不可用。

### 4.6 特征提取

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| doppler | 不可用 | 不可用 | 不可用 | 不可用 | 输出四维，当前三维 resize 直接报错 |
| entropy | 不可用 | 不可用 | 不可用 | 不可用 | 输出二维，当前 resize 直接报错 |
| ratio | 有限 | 可用 | 可用 | 有限 | 四个本地数据集都有至少两根天线，因此能运行；实数数据上只剩幅度比值 |
| decomposition | 不可用 | 可用 | 可用 | 不可用 | 实数输入时当前 CP 路径返回 dict，resize 需要 ndarray |
| conjugate_multiply | 不可用 | 可用 | 可用 | 不可用 | 要求复数 CSI；XRF55/ElderAL 是实数 |
| pca_fusion | 可用 | 条件 | 条件 | 可用 | Widar/Gait 与 z-score 组合会报错，与 min-max 或不归一化组合可以运行 |

#### Doppler 和 entropy

函数本身可以独立计算，但它们改变了数据结构：

```text
doppler → 四维频谱
entropy → 二维统计特征
```

当前 `resize_csi_to_fixed_length()` 写死接收 `(T,F,A)`，所以它们不能直接进入现有分类训练。

准确说法应是：

> 算法函数可计算，但当前分类训练管线不可用。

#### ratio

本地四个数据集的天线数都不少于 2，因此 ratio 可以生成三维输出并继续训练。

不过：

- Widar/Gait 有复数 CSI，天线比值能利用相对相位和幅度。
- XRF55/ElderAL 是实数，ratio 主要变成幅度比值，失去了原算法消除公共相位误差的主要意义。

因此 XRF55/ElderAL 上属于“可运行但意义有限”。

#### decomposition

在 Widar/Gait 的复数输入上，当前实现返回三维重建数组，可以继续训练。

在 XRF55/ElderAL 的实数输入上，默认 CP 分支返回包含 `weights`、`factors`、`reconstructed` 的 dict。后续 resize 调用 `sample.shape` 时会报错。

ElderAL 如果先执行 wavelet，数据可能被包装成复数，decomposition 会暂时变得“能运行”。但这依赖人工制造的复数类型，不建议将它视为稳定、合理的 ElderAL 组合。

#### conjugate multiply

它明确要求复数 CSI：

```text
Widar/Gait：可用
XRF55/ElderAL：直接 ValueError
```

同样，不能用 wavelet 把 ElderAL 包装成复数后就宣称获得了有效的共轭相乘特征。

#### PCA 与 z-score 的冲突

PCA 单独在四个数据集上都能生成三维数组。

但是 Widar/Gait 使用 z-score 时，当前 Processor 会把归一化推迟到 pipeline 结束，再要求结果仍是复数，以便生成：

```text
[带符号归一化幅度, 相位]
```

PCA 输出是实数，因此组合：

```text
Widar/Gait + z-score + pca_fusion
```

会报：

```text
return_phase_channels=True requires complex CSI data
```

以下组合可以运行：

```text
Widar/Gait + min-max + pca_fusion
Widar/Gait + no-normalize + pca_fusion
```

但 PCA 后的通道已经不再是原始幅度+相位表示，与普通预处理组合不完全公平，建议单独归为“特征表示实验”。

### 4.7 检测算法

| 算法 | XRF55 | Widar | Gait | ElderAL | 说明 |
|---|---|---|---|---|---|
| activity | 不可用 | 不可用 | 不可用 | 不可用 | 输出 `(T,)` 布尔序列，不是三维 CSI |
| change_point | 不可用 | 不可用 | 不可用 | 不可用 | 输出长度不固定的变点索引 |

检测函数本身能够执行，但不能接当前动作/用户分类模型。

它们回答的是：

```text
什么时候出现活动？
什么时候发生状态变化？
```

当前模型回答的是：

```text
这个完整样本属于哪个动作或用户？
```

任务定义不同。因此应记录为“当前分类训练管线不可用”，而不是认为检测算法数学实现坏了。

## 5. 为什么之前的六个预设能运行

六个预设主要使用：

```text
wavelet / butterworth / savgol
linear / polynomial / STC / robust
z-score / min-max
cubic interpolation
```

这些方法的共同特点是：

1. 输出基本保持三维 CSI；
2. 不要求额外的逐帧上下文；
3. 可以进入 resize 和 `CSIDataset`；
4. 没有使用 AGC、Doppler、entropy 或 detect。

所以预设能够训练是符合当前源码逻辑的。

但是预设跑通只能证明“这些特定组合没有报错”，不能证明：

- 所有26个注册算法都能训练；
- 每个算法在每个数据集上都有实际作用；
- 子载波坐标一定正确；
- `cubic30` 确实完成了插值；
- 准确率低一定是模型问题。

## 6. 能运行但可能无效或效果不好的情况

### 6.1 实数数据上的相位校准

适用数据集：

```text
XRF55
ElderAL
```

原因：输入没有真实虚部和相位。算法通常直接返回原输入，所以不报错，但没有完成有效相位校准。

记录建议：

```text
status = ok
effect_note = no-op_on_real_input
```

### 6.2 Widar/Gait 的子载波坐标没有自动传入

当前算法会运行，但 linear、polynomial、STC 和常规插值可能使用通用均匀坐标，而不是 IWL5300 的真实坐标。

这会导致：

- 相位趋势拟合不够准确；
- 插值横坐标不够准确；
- 结果可能有效，但不是算法预期的最佳实现。

它不是“不能用”，应标为：

```text
status = ok
effect_note = fallback_uniform_subcarrier_indices
```

### 6.3 `target_K == F`

算法直接返回输入副本，不报错、没有插值效果：

```text
status = ok
effect_note = identity_target_equals_input
```

### 6.4 bandpass 采样率不匹配

bandpass 默认认为：

```text
fs = 1000 Hz
```

Widar/Gait 的时间戳基本符合约 1000 Hz；XRF55 当前 `.npy` 不含真实时间戳；ElderAL 的时间戳间隔不规则且采样率较低。

采样率不正确时，滤波器仍会运行，但 0.5–50 Hz 的物理频段不再准确，可能保留或滤除错误的信号。

### 6.5 短序列上的滤波

Butterworth、bandpass、Savitzky–Golay 都需要一定的时间长度。

序列过短时，当前实现可能：

- 缩小窗口；
- 直接返回输入；
- 滤波效果很弱。

ElderAL 单个样本通常只有约32帧，应特别记录滤波前后差异，避免把“成功执行”误认为“确实完成了明显处理”。

### 6.6 插值改变输入维度

例如：

```text
30 → 64
30 → 15
512 → 64
```

输入维度变化后，部分模型的参数量和计算量也会变化。因此准确率变化不一定全部来自插值质量，也可能来自模型容量变化。

建议同时保存：

```text
input_shape
model_parameter_count
preprocess_time
train_time
```

## 7. 不修改源码时可以采用的处理办法

### 7.1 显式传入正确子载波索引

虽然 registry 没有正确传 Widar/Gait 的 dataset，但 `pipeline_steps` 中的普通参数仍会传给底层函数。

因此可以在测试配置中显式提供 IWL5300 子载波索引，不需要改源码：

```python
IWL5300_SUBCARRIERS = [
    -28, -26, -24, -22, -20, -18, -16, -14, -12, -10,
     -8,  -6,  -4,  -2,  -1,   1,   3,   5,   7,   9,
     11,  13,  15,  17,  19,  21,  23,  25,  27,  28,
]

pipeline_steps = {
    "calibrate": {
        "method": "stc",
        "subcarrier_indices": IWL5300_SUBCARRIERS,
    },
    "interpolate": {
        "method": "cubic",
        "target_K": 64,
        "subcarrier_indices": IWL5300_SUBCARRIERS,
    },
}
```

linear、polynomial、STC、linear/cubic/nearest interpolation 都可以采用这种方式。

robust 不依赖子载波坐标，不需要传。

### 7.2 为插值选择有效的 target

建议固定为：

| 数据集 | 原始 F | 上采样测试 | 降采样测试 |
|---|---:|---:|---:|
| XRF55 | 30 | 64 | 15 |
| Widar | 30 | 64 | 15 |
| Gait | 30 | 64 | 15 |
| ElderAL | 512 | 不建议上采样 | 64或30 |

其中：

- linear/cubic/nearest 可测试到64；
- decimate 只能使用比原始 F 小的目标；
- `target_K=30` 对三个30子载波数据集只作为“恒等对照”，不要当成有效插值。

### 7.3 显式传入 bandpass 的 fs

如果采样率已知，可以直接写进配置：

```python
{
    "denoise": {
        "method": "bandpass",
        "fs": 1000.0,
        "low_freq": 0.5,
        "high_freq": 50.0,
    }
}
```

Widar/Gait 可以使用约1000 Hz。XRF55 和 ElderAL 在采样率没有确认前，应标记为“参数不可靠”，而不是直接比较准确率后下结论。

### 7.4 对无法进入分类训练的算法保留失败记录

建议不要悄悄跳过，而是在 summary 中写明：

```text
agc                       → preprocess_error
doppler                   → incompatible_output_rank_4
entropy                   → incompatible_output_rank_2
activity                  → incompatible_output_rank_1
change_point              → variable_length_indices
decomposition+xrf55       → dict_output_on_real_input
decomposition+elderAL     → dict_output_on_real_input
conjugate_multiply+xrf55  → requires_complex_input
conjugate_multiply+elderAL→ requires_complex_input
pca_fusion+zscore+widar   → real_output_conflicts_with_phase_zscore
pca_fusion+zscore+gait    → real_output_conflicts_with_phase_zscore
```

这样既不修改源码，也能清楚回答“哪些当前不能用”。

## 8. 推荐的全量测试流程

### 第一阶段：方法级预检查

四个数据集分别测试全部26个方法，每个方法只取少量真实样本，不训练模型。

检查：

```text
是否抛异常
输出是否为 ndarray
输出维数
是否包含 NaN/Inf
输出是否与输入完全相同
预处理耗时
```

这一阶段必须保留失败算法，不能因为预期报错就不测试。

### 第二阶段：组合级预检查

当前适合直接做组合搜索的主流程是：

```text
denoise
→ outliers
→ calibrate
→ normalize
→ interpolate
```

建议候选项：

```text
denoise:
  none, wavelet, butterworth, savgol, bandpass, hampel

outliers:
  none, iqr, z-score

calibrate:
  none, linear, polynomial, stc, robust

normalize:
  none, z-score, min-max

interpolate for F=30:
  none, linear64, cubic64, nearest64, decimate15

interpolate for ElderAL:
  none, linear64, cubic64, nearest64, decimate64
```

如果保留实数数据上无效的相位校准作为对照，则每个数据集有：

```text
6 × 3 × 5 × 3 × 5 = 1350
```

种方法级配置，包括全 `none` 基线。

这一阶段仍然只做预处理，不训练模型。失败组合写入 summary，成功组合再进入训练。

### 第三阶段：分类训练

对第二阶段成功、输出为三维数组的组合训练固定模型。

训练失败时要区分：

| 失败位置 | 建议状态 |
|---|---|
| 算法函数报错 | `preprocess_error` |
| resize报错 | `representation_error` |
| CSIDataset报错 | `dataset_adapter_error` |
| 模型初始化或forward报错 | `model_compatibility_error` |
| 出现NaN | `numerical_error` |
| 正常训练完成 | `ok` |

只有前三类可以直接说明当前算法或表示不适合现有训练管线。模型报错可能只是某个模型与输入形状不兼容。

### 第四阶段：特征与检测算法

单独测试：

```text
ratio
decomposition
conjugate_multiply
pca_fusion
doppler
entropy
activity
change_point
```

其中 ratio、部分 decomposition、conjugate multiply、PCA 可以在满足条件时继续接分类模型。

Doppler、entropy 和 detect 应记录函数输出与运行时间，但在没有专用 adapter 的情况下，不进入当前分类模型训练。

## 9. 当前明确不能直接用于现有分类训练的项目

### 所有四个数据集都不能直接使用

```text
normalize: agc
extract_features: doppler
extract_features: entropy
detect: activity
detect: change_point
```

### 只在 XRF55、ElderAL 不能直接使用

```text
extract_features: decomposition
extract_features: conjugate_multiply
```

### 特定组合不能使用

```text
Widar + z-score + pca_fusion
Gait  + z-score + pca_fusion

decimate + target_K >= 原始F
ratio/conjugate_multiply + 单天线数据
```

本地四个数据集都至少有两根天线，因此 ratio 的单天线限制当前不会触发。

## 10. 当前能运行但应特别解释的项目

```text
XRF55 + 任意相位校准
ElderAL + 任意相位校准
Widar/Gait + 未显式提供真实子载波索引的 linear/polynomial/STC
Widar/Gait + 未显式提供真实子载波索引的 linear/cubic/nearest 插值
F=30 + target_K=30 的常规插值
XRF55/ElderAL + ratio
采样率不明确的数据集 + bandpass
短序列 + 高阶滤波或大窗口平滑
```

这些配置可以保留在实验中，但结论不能只写“成功”或“准确率低”，还需要写清算法是否真正改变了数据以及其物理前提是否成立。

## 11. 建议的结果字段

建议每条算法或组合至少记录：

```text
dataset
config_id
pipeline_steps
status
error_stage
error_type
error_message
input_shape
output_shape
input_dtype
output_dtype
is_complex_input
is_complex_output
contains_nan
contains_inf
output_equals_input
preprocess_seconds
model
seed
best_val_acc
test_acc
note
```

其中 `note` 可使用统一值：

```text
no_op_on_real_input
identity_target_equals_input
fallback_uniform_subcarrier_indices
requires_complex_input
requires_frame_agc
incompatible_output_rank
model_input_shape_changed
```

这样最后得到的不只是准确率排名，还能明确回答：

1. 哪些算法能进入现有训练流程；
2. 哪些算法在什么数据集上会报错；
3. 哪些算法虽然能跑但没有实际处理效果；
4. 哪些算法可能因为参数或数据前提不匹配而效果较差；
5. 哪些失败来自算法，哪些失败只是模型兼容性问题。
