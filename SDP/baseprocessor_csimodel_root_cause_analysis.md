# 四个数据集源码修改记录

本文记录当前已经落到本地 WSDP 源码里的修改。重点是：数据怎么读、label/group 怎么定、怎么划分、算法处理哪里变了，以及这些修改是否会影响其他数据集。

源码根目录：

```text
SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/src/wsdp
```

## 当前最终修改摘要

### 1. 在已有配置模块中加入数据集策略

**文件**

```text
src/wsdp/configs/__init__.py
```

**改了什么**

- 在已有文件里加入 `AMPLITUDE_PRIMARY_DATASETS = {"xrf55"}`。
- 在已有文件里加入 `is_amplitude_primary_dataset(dataset)`。
- 在已有文件里加入 `real_if_negligible_imaginary(data, dataset=None)`。

**为什么这样改**

XRF55 在当前处理链路里属于“以幅度为主”的数据。部分算法会把实数幅度包装成 complex，但虚部几乎为 0。这里把这种判断集中在已有配置模块里，避免在多个源码文件里到处写死 `dataset == "xrf55"`，同时也满足“不新增 py 文件”的要求。

**影响范围**

目前只有 XRF55 会进入这个策略。Widar、Gait、ElderAL 不在 `AMPLITUDE_PRIMARY_DATASETS` 里，因此不会被这套幅度数据修正逻辑影响。

### 2. XRF55 不再拆成 3 个 Rx 样本

**文件**

```text
src/wsdp/readers/xrf_reader.py
src/wsdp/processors/base_processor.py
```

**改了什么**

- 一个 `.npy` 文件保留为一个完整样本。
- 样本形状从拆分后的 `(1000, 30, 3)` 改为完整链路的 `(1000, 30, 9)`。
- `9 = 3 Rx * 3 antenna`。

**为什么这样改**

XRF55 一个文件本来就包含完整的 3 个接收端信息。拆开以后，模型每次只看到一部分链路，信息被削弱。

**影响范围**

只改 XRF55 reader 和 XRF55 文件名解析逻辑，不影响 Widar、Gait、ElderAL。

### 3. XRF55 改成 repetition 划分

**文件**

```text
src/wsdp/core.py
```

**改了什么**

XRF55 使用固定 repetition 划分：

```text
repetition 1-12  -> train
repetition 13-16 -> valid
repetition 17-20 -> test
```

label 仍然是 `action_id`，任务仍然是动作识别。

**为什么这样改**

XRF55 原项目更接近按动作重复次数划分，而不是按用户划分。按用户划分会把问题变成更难的跨用户泛化实验，不适合作为当前预设算法组合的基础对比。

**影响范围**

`core.py` 里只有 `dataset == "xrf55"` 时才进入 `_create_xrf55_repetition_split()`。其他数据集仍走自己的 group split。

### 4. XRF55 归一化改成训练集统一归一化

**文件**

```text
src/wsdp/processors/configurable_processor.py
src/wsdp/core.py
```

**改了什么**

- 如果 XRF55 的 preset 里有 `z-score` 或 `min-max`，先不在每个样本内部单独归一化。
- 划分好 train/valid/test 后，只用 train 计算归一化参数。
- 再把同一套参数应用到 train/valid/test。

**为什么这样改**

之前相当于“每个样本自己洗自己”，幅度强弱差异被洗掉。XRF55 识别动作时，幅度变化本身很有用，所以要改成“训练集统一标准”，让 valid/test 用同一把尺子。

**影响范围**

只对 `is_amplitude_primary_dataset(dataset)` 为真的数据集生效，目前也就是 XRF55。Widar、Gait、ElderAL 不受影响。

### 5. pipeline 算法调用参数更明确

**文件**

```text
src/wsdp/algorithms/registry.py
```

**改了什么**

- `execute_pipeline()` 支持传 `dataset`。
- 用显式白名单决定哪些算法需要 `method`，哪些算法需要 `dataset`。
- 不再靠 `inspect.signature()` 临时猜参数。
- 每一步算法后，如果是幅度主数据集且输出是假 complex，就转回 real。

**为什么这样改**

Reviewer 不希望靠函数签名猜参数，这样太隐式。显式白名单更稳，也更容易审查。

**影响范围**

只把参数传给明确列出的算法。未列入的算法调用方式保持原样。

### 6. CSIDataset 增加 signed real 保护开关

**文件**

```text
src/wsdp/datasets/CSIDataset.py
```

**改了什么**

- 增加 `preserve_real_sign=False`。
- 默认仍然保持旧逻辑：复数或实数都按原来的幅度输入处理。
- 只有 XRF55 这种需要保留 z-score 正负号的场景，才显式打开。

**为什么这样改**

XRF55 做 z-score 后，正负号有意义。如果最后再 `abs()`，`-2` 和 `+2` 会都变成 `2`，信息丢失。

**影响范围**

默认值是 `False`，所以不主动影响 Widar、Gait、ElderAL。

### 7. Widar 和 Gait 的 group 语义写入源码和文档

**文件**

```text
src/wsdp/processors/base_processor.py
docs/datasets/widar.md
docs/datasets/gait.md
```

**改了什么**

- Widar:
  - label = `gesture_type`
  - group = `position_id * 1000 + orientation_id * 100 + receiver_id`
- Gait:
  - label = `user_id`
  - group = `track_id * 100 + receiver_id`

**为什么这样改**

把“模型预测什么”和“数据按什么条件划分”分开，避免 label/group 混在一起。

**影响范围**

Widar 只在 `dataset == "widar"` 分支生效，Gait 只在 `dataset == "gait"` 分支生效。

### 8. 测试入口已经精简

当前每个数据集只保留两个入口：

| 数据集 | 单次测试 | 全量 preset/model 测试 |
|---|---|---|
| XRF55 | `test_xrf55/pipline_xrf55_repetition.py` | `test_xrf55/full_test_presets_models_xrf55.py` |
| Widar | `test_wider/pipline_wider_v2.py` | `test_wider/full_test_presets_models_widar.py` |
| Gait | `test_gait/pipline_gait.py` | `test_gait/full_test_presets_models_gait.py` |
| ElderAL | `test_elderAL/pipline_elderAL.py` | `test_elderAL/full_test_presets_models_elderAL.py` |

这些脚本现在都直接调用源码 `wsdp.pipeline()`，不再维护一套自写训练/划分/处理流程。

数据路径统一改为：

```text
/home/test/bupt_hjk/sdp_dataset
```

结果目录已经整理为：

```text
test_xrf55/result/preset_tests
test_wider/result/preset_tests
test_gait/result/preset_tests
test_elderAL/result/preset_tests
```

旧实验脚本和旧结果目录已经删除，只保留最终认可的一轮结果。

## 当前数据集语义

| 数据集 | label | group | 划分逻辑 |
|---|---|---|---|
| XRF55 | `action_id` | `repetition_id` | 固定 repetition：1-12 train，13-16 valid，17-20 test |
| Widar | `gesture_type` | `position_id * 1000 + orientation_id * 100 + receiver_id` | `GroupShuffleSplit` |
| Gait | `user_id` | `track_id * 100 + receiver_id` | `GroupShuffleSplit` |
| ElderAL | `action_id` | `position_id` | `GroupShuffleSplit` |

---

# 1. XRF55

## 第一次修改：不再把一个 `.npy` 拆成 3 个 Rx 样本

**源码位置**

- `src/wsdp/readers/xrf_reader.py`
  - `_read_npy()`，约第 160-205 行。
- `src/wsdp/processors/base_processor.py`
  - `_parse_file_info_from_filename()`，约第 278-294 行。
  - `_selector()`，约第 358-365 行。
- `src/wsdp/core.py`
  - `_create_data_split()`，约第 98-119 行。
  - `_create_xrf55_repetition_split()`，约第 162-233 行。

**改了什么**

1. 读取逻辑：
   - 之前：一个 `.npy` 文件拆成 3 个 `CSIData` 样本，每个样本只包含 1 个 Rx。
   - 现在：一个 `.npy` 文件保留为 1 个完整 `CSIData` 样本，包含 3 个 Rx。

2. shape 变化：

```text
原始 .npy：
  raw shape 通常等价于 (270, 1000)
  270 = 3 Rx * 30 subcarrier * 3 antenna
  1000 = 时间长度

源码内部 reshape：
  (3, 30, 3, 1000)
  3    = Rx 数
  30   = 子载波数
  3    = 每个 Rx 的天线数
  1000 = 时间长度

旧逻辑拆分后，一个 .npy -> 3 个样本：
  单个样本 shape = (1000, 30, 3)
  1000 = 时间
  30   = 子载波
  3    = 当前这个 Rx 下的 3 根天线

新逻辑不拆分，一个 .npy -> 1 个样本：
  单个样本 shape = (1000, 30, 9)
  1000 = 时间
  30   = 子载波
  9    = 3 Rx * 3 antenna，完整链路一起保留
```

3. 文件名解析：
   - 之前只取 `user_id/action_id`。
   - 现在解析 `user_id/action_id/repetition_id`。

4. label/group：
   - label 仍然是 `action_id`，也就是 55 类动作识别任务不变。
   - group 改为 `repetition_id`。

5. 划分：

```text
repetition 1-12  -> train
repetition 13-16 -> valid
repetition 17-20 -> test
```

**为什么这样改**

XRF55 一个 `.npy` 本来就包含完整的 3 Rx × 3 Ant × 30 Subcarrier 信息。拆成 3 份后，每个样本只看到一部分链路，信息被削弱。并且 XRF55 更适合按动作重复次数划分，而不是按 user_id 划分。

**测试情况**

旧拆分 + repetition 结果较低：

```text
结果文件：test_xrf55/result/repetition_all_presets_models_summary.csv
最好组合：activity_detection + cnn1dmodel
test acc：0.3586
结果目录：test_xrf55/result/repetition+activity_detection+cnn1dmodel
```

no-split + repetition 后提升：

```text
结果文件：test_xrf55/result/source_nosplit_repetition_all_presets_models_summary.csv
最好组合：high_quality + mlpmodel
test acc：0.6636
结果目录：test_xrf55/result/source_nosplit_repetition+high_quality+mlpmodel
```

## 第二次修改：处理 XRF55 幅度数据被算法变成“假复数”的问题

**源码位置**

- `src/wsdp/processors/base_processor.py`
  - `_process_single_csi()`，约第 234-240 行。
- `src/wsdp/algorithms/registry.py`
  - `execute_pipeline()`，约第 377-445 行。
- `src/wsdp/datasets/CSIDataset.py`
  - `CSIDataset.__init__()`，约第 6-31 行。

**改了什么**

1. `BaseProcessor`：
   - XRF55 做完 `wavelet_denoise_csi()` 后，如果输出是 complex，但虚部几乎为 0，就转回 real。

2. `execute_pipeline()`：
   - 增加 `dataset=None` 参数。
   - 执行每一步算法时，如果是 XRF55，且输出是“虚部几乎为 0 的 complex”，就转回 real。
   - 修复 `method` 没有传给算法函数的问题，保证 `normalize: min-max` 真的走 `min-max`，不会误走默认 `z-score`。

3. `CSIDataset`：
   - 增加 `preserve_real_sign=False`。
   - 默认行为仍是原来的 `np.abs()`。
   - 只有显式打开时才保留 real/z-score 数据的正负号。

**为什么这样改**

当前本地 XRF55 `.npy` 更像幅度数据，不是真正带相位的复数 CSI。部分去噪/归一化算法会把它包装成 complex，但虚部实际接近 0。继续按复数处理容易导致后面又 `abs()`，把 z-score 的正负号折叠掉。

**测试情况**

这一步修复了数据形态和 `min-max` 参数问题，但单独这一步还不能让预设组合达到理想精度。后面继续定位到“单样本归一化会洗掉幅度差异”。

## 第三次修改：XRF55 的归一化改为训练集统一归一化

**源码位置**

- `src/wsdp/processors/configurable_processor.py`
  - `_process_single_csi_configurable()`，约第 63-78 行。
- `src/wsdp/core.py`
  - `_create_data_split()`，约第 98-119 行。
  - `_create_xrf55_repetition_split()`，约第 204-231 行。
  - `pipeline()` 创建 `CSIDataset`，约第 452-469 行。

**测试入口同步修改**

- `test_xrf55/full_test_presets_models_xrf55.py`
  - `make_loaders()` 调用 `_create_data_split(..., dataset="xrf55", pipeline_steps=pipeline_steps)`。
- `test_xrf55/pipline_xrf55_repetition.py`
- `test_xrf55/pipline_xrf55_repetition_nosplit.py`

**改了什么**

1. `ConfigurableProcessor`：
   - 如果是 XRF55，且 preset 里有 `z-score/min-max`，先从单样本 pipeline 里跳过 `normalize`。

2. `core.py`：
   - 先按 repetition 切出 train/valid/test。
   - 再只用 train 计算归一化参数。
   - 最后把同一套参数应用到 train/valid/test。

3. `CSIDataset`：
   - 只有 XRF55 且 z-score 时，才传 `preserve_real_sign=True`。
   - 其他情况保持默认 `abs()`。

**为什么这样改**

XRF55 的动作差异有一部分体现在“信号强弱”上。原来的预设归一化是每个样本自己归一化自己，相当于每个样本都单独洗了一遍，把样本之间真实的强弱差异洗掉了。训练集统一归一化相当于所有样本共用一把尺子，更适合 XRF55。

**测试情况**

单独验证脚本效果明显提升：

```text
脚本：test_xrf55/pipline_xrf55_repetition_trainnorm.py
结果目录：test_xrf55/result/train_global_norm+source_nosplit+repetition+high_quality+mlpmodel
组合：high_quality + mlpmodel
best val acc：90.61%
test acc：0.8424
```

当前全量测试 summary：

```text
结果文件：test_xrf55/result/train_norm_nosplit_repetition_all_presets_models_summary.csv
当前最好组合：fast + cnn1dmodel
best val acc：90.61%
test acc：0.8470
结果目录：test_xrf55/result/train_norm_nosplit_repetition+fast+cnn1dmodel
```

---

# 2. Widar

## 修改：把 group 从 user_id 改成采集条件 group

**源码位置**

- `src/wsdp/processors/base_processor.py`
  - `_parse_file_info_from_filename()`，约第 247-256 行。
  - `_selector()`，约第 347-351 行。
- `src/wsdp/core.py`
  - cache key 标记，约第 387-391 行。

**改了什么**

1. label 保持为 `gesture_type`。
2. group 从旧的 `user_id` 改为：

```python
group = position_id * 1000 + orientation_id * 100 + receiver_id
```

3. cache key 增加 Widar 划分版本标记：

```text
widar_condition_position_orientation_receiver
```

**为什么这样改**

当前 Widar 测试目标是手势识别。我们希望训练、验证、测试在位置/朝向/接收机条件上分开，而不是按用户分开。这个逻辑之前只在测试脚本里，现在同步到了源码。

**测试情况**

```text
结果文件：test_wider/result/condition_v2_all_presets_models_summary.csv
最好组合：gesture_recognition + mlpmodel
best val acc：70.24%
test acc：0.6869
结果目录：test_wider/result/condition_v2+gesture_recognition+mlpmodel
```

---

# 3. Gait

## 修改：明确任务是 User ID 识别

**源码位置**

- `src/wsdp/processors/base_processor.py`
  - `_parse_file_info_from_filename()`，约第 260-274 行。
  - `_selector()`，约第 352-357 行。

**改了什么**

1. 文件名解析为：

```python
user_id, track_id, repetition_id, receiver_id
```

2. label/group 改为：

```python
label = user_id
group = track_id * 100 + receiver_id
```

3. 原来把 activity/repetition 当语义的代码已经注释保留。

**为什么这样改**

Gait 当前任务是识别用户身份，不是识别动作类别。group 用 track + receiver，是为了让训练/验证/测试在轨迹和接收机条件上分开。

**测试情况**

```text
结果文件：test_gait/result/user_id_v2_all_presets_models_summary.csv
最好组合：activity_detection + resnet2d
best val acc：86.19%
test acc：0.9289
结果目录：test_gait/result/user_id_v2+activity_detection+resnet2d
```

---

# 4. ElderAL

## 第一次修改：确认 label/group 语义

**源码位置**

- `src/wsdp/processors/base_processor.py`
  - `_parse_file_info_from_filename()`，约第 298-304 行。
  - `_selector()`，约第 366-368 行。

**当前逻辑**

```python
label = action_id
group = position_id
```

**为什么这样处理**

ElderAL 当前任务是动作识别，位置作为 group 做划分。这个语义和当前测试目标一致。

## 第二次修改：修复 Butterworth 短序列报错

**源码位置**

- `src/wsdp/algorithms/denoising_butterworth.py`
  - `butterworth_denoise()`，约第 42-71 行。

**改了什么**

短序列保护从：

```python
if T < min_len:
```

改成：

```python
if T <= min_len:
```

**为什么这样改**

ElderAL 有些样本长度刚好等于 `scipy.signal.filtfilt()` 的 `padlen`。这种情况下 scipy 要求输入长度必须大于 `padlen`，否则会报：

```text
The length of the input vector x must be greater than padlen
```

现在长度不够安全滤波时直接原样返回；正常长度样本仍然照常滤波。

**测试情况**

```text
结果文件：test_elderAL/result/source_action_position_all_presets_models_summary.csv
最好组合：high_quality + csitime
best val acc：92.51%
test acc：0.9148
结果目录：test_elderAL/result/source_action_position+high_quality+csitime
```

---

# 5. 公共算法层修改

## `src/wsdp/algorithms/registry.py`

**位置**

- `execute_pipeline()`，约第 377-445 行。

**改了什么**

- 增加 `dataset=None` 参数。
- 根据算法函数签名自动传 `method`，修复 `min-max` 不生效的问题。
- 只在 `dataset == "xrf55"` 时，把虚部接近 0 的假 complex 转回 real。

**影响范围**

- `method` 参数修复是公共 bug fix。
- 假 complex 转 real 只影响 XRF55。
- Widar/Gait 真复数 CSI 不进入 XRF55 分支。
- ElderAL 不进入 XRF55 分支。

## `src/wsdp/datasets/CSIDataset.py`

**位置**

- `CSIDataset.__init__()`，约第 6-31 行。

**改了什么**

- 增加 `preserve_real_sign=False`。
- 默认仍然 `np.abs()`，所以不影响原始默认行为。
- 只有 XRF55 + z-score 场景显式传入 `True` 时，才保留 z-score 的正负号。

---

# 6. 当前结论

目前四个数据集需要落到源码里的核心修改已经完成：

- XRF55：不拆 Rx、按 repetition 划分、假复数处理、训练集统一归一化已完成。
- Widar：condition group 划分已同步到源码。
- Gait：User ID 识别语义已同步到源码。
- ElderAL：action/position 语义确认，Butterworth 短序列报错已修复。

后续如果继续优化，主要是继续跑全量组合、更新 summary，不需要再改这些核心源码，除非新的训练结果暴露新的报错或异常。
