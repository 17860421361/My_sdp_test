# 四数据集算法组合测试重设计方案

> 本文档是对 `recommended_algorithm_combinations.md` 的补充和修订建议，不替代、不修改原文。
>
> 目标不是立即扩大组合数量，而是先保证每一种算法在进入模型前确实生效，删除等价组合，再以可复现、可断点续跑的方式完成四个数据集与全部模型的测试。

## 1. 总体结论

原推荐表可以作为第一轮讨论稿，但不适合直接启动正式全量实验，主要原因如下：

1. 名义上的不同组合不一定产生不同的模型输入。
2. 当前模型默认只接收 CSI 幅值，很多相位校准结果会在 `abs()` 时消失。
3. XRF55 的 z-score/min-max 会被处理器主动删除。
4. Widar 和 Gait 原始数据都是 30 个子载波，`cubic30` 是恒等操作。
5. ElderAL 是 amplitude-only 数据，相位校准是空操作。
6. z-score 的负值在进入模型前被取绝对值，实际变成了 `abs(z-score)`。
7. 当前批量脚本跳过了 4 个注册模型，因此还不是“全部模型”测试。

因此，正式测试前需要先明确模型输入表示，并将实验拆成 amplitude-only 和 amplitude+phase 两条支线。

## 2. 当前可用算法范围

源码注册表中的算法包括：

| Category | Methods |
|---|---|
| denoise | wavelet, butterworth, savgol, bandpass, hampel |
| outliers | iqr, z-score |
| calibrate | linear, polynomial, stc, robust |
| normalize | z-score, min-max, agc |
| interpolate | linear, cubic, nearest, decimate |
| extract_features | doppler, entropy, ratio, decomposition, conjugate_multiply, pca_fusion |
| detect | activity, change_point |

源码当前注册了 19 个模型。现有批量脚本跳过：

```text
efficientnetcsi
visiontransformercsi
graphneuralcsi
mambacsi
```

所以现有脚本实际只测试 15 个模型。

## 3. 原 126 组中的等价和无效组合

### 3.1 XRF55

`ConfigurableProcessor` 会删除 XRF55 的 z-score/min-max：

```python
if (
    dataset == "xrf55"
    and normalize_step.get("method") in {"z-score", "min-max"}
):
    effective_pipeline_steps = {
        key: value
        for key, value in pipeline_steps.items()
        if key != "normalize"
    }
```

因此，配置名称中虽然包含 normalization，预处理阶段实际上没有执行它。

当前本地 XRF55 数据为非负 amplitude `.npy`。在不使用相位特征和非平凡复数插值时，linear/robust 等相位校准也不会改变最终的幅值模型输入。

原来的 18 组最多形成约 6 种输入，实际很可能只有 3 种主要输入，即三种 denoise 的结果。

### 3.2 Widar

Widar 原始子载波数为 30：

```text
none == cubic(target_K=30)
```

没有非平凡复数插值、相位特征或 amplitude+phase 表示时，不同 phase calibration 在进入模型前会被 `abs()` 消除。

`cubic64` 会对校准后的复数 CSI 插值，因此不同校准方法仍可能通过插值影响最终幅值。

原来的 54 组最多约有 24 组能够形成不同输入。

### 3.3 Gait

Gait 同样是 30 个子载波，因此 `cubic30` 与不插值完全相同。

当前表中没有 `cubic64` 等非恒等插值，且模型只使用幅值，因此 linear、polynomial、STC 对模型基本不可见。

原来的 36 组实际约等价于：

```text
3 denoise × 2 normalize = 6
```

### 3.4 ElderAL

ElderAL reader 明确将数据定义为 amplitude-only。linear、STC、robust 等相位校准都会返回原数据。

原来的 18 组实际约等价于：

```text
3 denoise × 2 normalize = 6
```

### 3.5 粗略汇总

| Dataset | 原推荐数量 | 当前实现下最多约有区别的输入 |
|---|---:|---:|
| xrf55 | 18 | 3～6 |
| widar | 54 | 24 |
| gait | 36 | 6 |
| elderAL | 18 | 6 |
| Total | 126 | 不超过约 42 |

这里统计的是“可能产生不同模型输入的配置”，不是最终建议的组合数量。修复模型输入语义后，组合空间需要重新生成。

## 4. 必须先确定的模型输入表示

### 4.1 支线 A：amplitude-only

适用范围：

- XRF55
- ElderAL
- Widar/Gait 的幅值基线实验
- 只支持实数幅值输入的模型

建议处理语义：

```text
raw CSI
  -> optional AGC
  -> optional outlier removal
  -> optional denoise
  -> amplitude extraction
  -> optional frequency resample
  -> optional amplitude normalization
  -> model
```

该支线不测试 phase calibration，因为最终只保留幅值时，相位校准没有独立意义。

z-score 输出必须保留正负号，不能再统一执行 `abs(z-score)`。

### 4.2 支线 B：amplitude+phase

适用范围：

- Widar
- Gait
- 其他具有可靠复数 CSI 的数据

建议处理语义：

```text
complex CSI
  -> optional AGC
  -> optional outlier removal
  -> optional denoise
  -> optional phase calibration
  -> optional complex frequency resample
  -> amplitude = abs(CSI)
  -> phase = angle(CSI)
  -> normalize amplitude without destroying its sign
  -> concatenate [amplitude, phase]
  -> model
```

该支线才用于公平比较：

```text
none
linear
polynomial
stc
robust
```

不建议继续用“负的标准化幅值乘以复相位”的方式表达 z-score。幅值和相位应作为两个显式实数通道保存。

### 4.3 支线 C：特征与检测

以下算法会改变数据形态或任务含义：

```text
doppler
entropy
ratio
decomposition
conjugate_multiply
pca_fusion
activity detection
change-point detection
```

它们不应直接加入普通 `(T, F, A)` CSI 的全模型笛卡尔积，而应建立单独的输入适配器：

- Doppler：频谱/时频图模型
- Entropy：向量或传统分类模型
- Ratio、conjugate_multiply：复数或 amplitude+phase 模型
- Decomposition：明确使用重建 CSI 还是低维 factors
- Detection：作为检测任务，不与活动类别分类混为一组

## 5. 基础方法和参数建议

### 5.1 Denoise

| ID | Configuration | 备注 |
|---|---|---|
| none | 不执行 denoise | 必须保留的基线 |
| wavelet | `{"method": "wavelet"}` | 可后续比较 visu/bayes |
| butterworth | `{"method": "butterworth", "order": 4, "cutoff": 0.25}` | cutoff 是归一化频率 |
| savgol | `{"method": "savgol", "window_length": 11, "polyorder": 3}` | 窗长应结合采样率解释 |
| hampel | `{"method": "hampel", "window_size": 5, "n_sigma": 3.0}` | 与 outliers 有功能重叠 |
| bandpass | 数据集专用参数 | 必须先确定真实采样率 `fs` |

`bandpass` 不应在不知道采样率时使用默认 `fs=1000`。不同数据集的相同 window/cutoff 也不一定代表相同的物理时间或频率范围。

### 5.2 Outliers

| ID | Configuration |
|---|---|
| none | 不执行 |
| iqr | `{"method": "iqr", "factor": 1.5}` |
| z-score | `{"method": "z-score", "factor": 3.0}` |

第一轮不建议同时组合 Hampel denoise 和 outliers，避免两个异常值算法重复处理。等单算法筛选完成后，再决定是否测试交互。

### 5.3 Phase calibration

| ID | Configuration |
|---|---|
| none | 不执行 |
| linear | `{"method": "linear"}` |
| polynomial | `{"method": "polynomial", "degree": 2}` |
| stc | `{"method": "stc"}` |
| robust | `{"method": "robust"}` |

这些方法只放入 amplitude+phase 支线，或者放在确实会使用复数相位的特征提取之前。

### 5.4 Normalization

| ID | Configuration |
|---|---|
| none | 不执行 |
| z-score | `{"method": "z-score"}` |
| min-max | `{"method": "min-max"}` |

要求：

1. XRF55 不再静默删除 normalization。
2. z-score 的负值必须原样送入模型。
3. amplitude+phase 支线只归一化 amplitude 通道，不把 phase 当幅值归一化。
4. AGC 不作为普通 normalization 与所有数据集混跑。

### 5.5 AGC

AGC 需要每个时间帧的 `agc_values`。当前处理器只向算法传入 CSI 数组，没有传递 frame metadata，所以现阶段无法直接加入组合。

建议：

- 只对确实包含 AGC 元数据的 Intel IWL5300 数据启用。
- 优先考虑 Widar/Gait。
- 在 outlier、denoise 和 normalization 之前执行。
- XRF55 `.npy` 和 ElderAL 不强行加入 AGC。

### 5.6 Frequency resample

对于原始 `F=30` 的 XRF55、Widar、Gait：

| ID | Configuration | 含义 |
|---|---|---|
| none | 不执行 | 原始 30 子载波 |
| decimate15 | `{"method": "decimate", "target_K": 15}` | 抗混叠降采样 |
| linear64 | `{"method": "linear", "target_K": 64}` | 线性上采样 |
| cubic64 | `{"method": "cubic", "target_K": 64}` | 三次上采样 |
| nearest64 | `{"method": "nearest", "target_K": 64}` | 最近邻对照 |

必须删除：

```text
linear30
cubic30
nearest30
```

因为目标子载波数与原始数相同，函数会直接返回输入副本。

对于 ElderAL，应先检查所有文件实际的 `F` 是否一致。如果为 512，优先测试：

```text
none
decimate128
decimate64
```

不建议直接用无抗混叠的 cubic 插值完成 512→64 降采样。

## 6. 推荐的分阶段实验流程

### 阶段 0：语义和兼容性验证

在任何正式训练前，每个数据集选取少量固定样本进行预检查：

1. 检查输入 dtype、是否复数、`T/F/A`。
2. 检查输出 shape、dtype、NaN、Inf。
3. 检查 normalization 是否真正改变数据。
4. 检查 phase calibration 是否真正改变 phase。
5. 检查 amplitude-only 支线是否完全不依赖 phase calibration。
6. 检查 `target_K` 是否真的改变频率维度。
7. 对每个模型执行一次小 batch forward。
8. 记录处理前后的统计量和内容 hash。

预检查失败的组合不得进入正式队列。

### 阶段 1：单算法筛选

使用固定 split、固定 seed 和 2～3 个代表模型测试单个算法的边际效果。

建议代表模型覆盖不同结构，例如：

```text
mlpmodel
cnn1dmodel
csitime
```

每次只改变一个因素：

```text
raw baseline
baseline + one denoise
baseline + one outlier method
baseline + one normalization
baseline + one resample method
phase baseline + one calibration method
```

这一阶段使用较少 epoch，只根据 validation 指标筛选。

### 阶段 2：有限交互组合

每个数据集从阶段 1 中选择：

- top 2 denoise，加上 none
- top 1 outlier，加上 none
- normalization 保留 none、z-score、min-max
- top 2 resample，加上 none

amplitude-only 候选空间：

```text
3 denoise × 2 outliers × 3 normalize × 3 resample = 54
```

ElderAL 如果只保留一种降采样，则为：

```text
3 × 2 × 3 × 2 = 36
```

phase calibration 不再与所有因素做完整笛卡尔积。对 Widar/Gait，先固定阶段 2 中最好的 amplitude pipeline，再比较：

```text
5 calibrate × 1～2 resample = 5～10
```

这样既能观察校准差异，也能避免无意义地把 phase calibration 乘到所有幅值组合中。

### 阶段 3：全部模型验证

每个数据集选择：

- validation 最好的若干 amplitude-only 组合
- 至少一个 raw/none 基线
- 至少一个结构上不同但表现稍弱的多样性候选
- Widar/Gait 最好的若干 amplitude+phase 组合

然后再扩展到全部 19 个模型。

这一阶段要求：

- 每个模型有独立 batch size。
- OOM 时自动减小 batch size，而不是直接永久跳过。
- 记录模型与输入 shape 的兼容性。
- 使用至少 3 个 seed。
- 汇报 mean、std，而不只汇报单次最高值。

### 阶段 4：最终测试集评估

组合筛选只能使用 validation set。

不应在每个候选组合上反复查看 test accuracy 后再选择最好组合，否则 test set 会参与模型选择。

正确流程：

```text
train/validation 选择算法和模型
  -> 锁定配置
  -> 对锁定配置运行最终 test
```

## 7. 自动去重策略

### 7.1 静态去重

生成组合时直接应用规则：

1. `target_K == original_F` 的 interpolate 删除。
2. amplitude-only 数据删除 calibrate。
3. amplitude-only 支线删除 calibrate。
4. 没有 AGC metadata 的数据删除 agc。
5. `decimate.target_K >= original_F` 删除。
6. bandpass 缺少可信 `fs` 时删除。
7. feature/detect 不进入普通 CSI 模型队列。

### 7.2 运行时去重

对完整预处理结果生成 fingerprint：

```text
dataset
split_id
representation_mode
output shape
output dtype
labels/groups
SHA-256(processed arrays)
```

如果两个配置产生相同 fingerprint：

- 只保留一个 canonical combination。
- 其他组合记录为 `equivalent`。
- 保存 `equivalent_to`，而不是再次训练全部模型。

不能只根据组合名称判断是否不同。

## 8. 组合和结果记录格式

每个组合应保存规范化 JSON，例如：

```json
{
  "dataset": "widar",
  "representation": "amplitude_phase",
  "steps": {
    "outliers": null,
    "denoise": {
      "method": "butterworth",
      "order": 4,
      "cutoff": 0.25
    },
    "calibrate": {
      "method": "stc"
    },
    "resample": {
      "method": "cubic",
      "target_K": 64
    },
    "normalize_amplitude": {
      "method": "z-score"
    }
  }
}
```

建议 summary 至少记录：

| Field | 说明 |
|---|---|
| dataset | 数据集 |
| combination_id | 规范组合 ID |
| representation | amplitude / amplitude_phase / feature |
| pipeline_json | 完整配置 |
| preprocess_hash | 预处理结果 hash |
| equivalent_to | 等价组合 ID |
| model | 模型 |
| seed | 随机种子 |
| split_id | 数据划分标识 |
| input_shape | 模型实际输入 shape |
| batch_size | 实际 batch size |
| status | ok / failed / oom / incompatible / equivalent |
| best_val_acc | 最佳验证准确率 |
| best_val_macro_f1 | 最佳验证 macro-F1 |
| test_acc | 最终阶段测试准确率 |
| test_macro_f1 | 最终阶段测试 macro-F1 |
| preprocess_seconds | 预处理时间 |
| train_seconds | 训练时间 |
| error | 错误信息 |
| source_version | 源码版本或 commit |

## 9. 断点续跑和错误处理要求

当前逻辑会把 CSV 中任何已有记录都视为完成，包括 `failed` 和 `skipped`。新脚本应改为：

```text
status=ok          -> 默认不重跑
status=equivalent  -> 不重跑，复用 canonical result
status=failed      -> 允许重试
status=oom         -> 减小 batch size 后重试
status=incompatible -> 明确记录兼容性原因
```

预处理也必须放入组合级异常捕获。一个组合失败后，应记录错误并继续下一个组合，不能中断整个数据集任务。

## 10. 运行前还需处理的工程问题

1. 当前本地数据目录为 `widar_common3` 和 `Gait_Dataset`，脚本配置使用 `widar` 和 `gait`，需要统一路径。
2. 对全部 19 个模型建立 model-specific batch size。
3. 对会显著扩大输入的 `target_K=64` 单独估算显存。
4. 预处理结果按 dataset+combination 缓存，供多个模型复用。
5. 固定 split，并保存每个 split 的样本索引或 group 列表。
6. 完整设置 Python、NumPy、PyTorch、CUDA 的随机性和确定性选项。
7. 不以 test accuracy 作为候选组合排序依据。

## 11. 推荐实施顺序

```text
1. 修复输入表示和 normalization 语义
2. 增加 amplitude / amplitude_phase 两种明确模式
3. 增加组合合法性检查和静态去重
4. 增加预处理 fingerprint 去重
5. 完成单算法小规模筛选
6. 生成有限交互组合
7. 用代表模型筛选
8. 候选组合扩展到全部模型
9. 多 seed 验证
10. 锁定配置后执行最终 test
```

该顺序的核心原则是：先证明算法真正改变了模型输入，再训练模型；先用 validation 筛选，再使用 test；先消除重复组合，再扩大模型规模。
