# 当前对话导出

导出时间：2026-06-17

## 重要说明

这份文档是当前会话的可恢复导出。

由于这条对话很长，系统中途发生过上下文压缩。压缩前的部分助手回复已经不再以逐字原文形式出现在当前上下文里，因此我不能保证把压缩前每一句助手回答逐字还原。为了不编造内容，本文件采用下面的方式记录：

- 用户问题：按当前上下文中可见内容尽量完整记录。
- 助手回答：压缩前部分用当前上下文保留的摘要和已生成文档内容归纳；最近几轮可见回答尽量逐字保留。
- 文件、脚本、诊断结果：按当前工作区和会话摘要中可确认的信息记录。

也就是说，这不是“聊天系统原始日志逐字备份”，而是“当前上下文可恢复范围内的完整整理版”。

---

## 一、最初的 XRF55 低准确率问题

### 用户问题

用户最初提出：

> 我在 `full_pipeline_xrf55.py` 上，跑 xrf55 数据集的结果是在测试集上准确率特别低，约为个位数。`baseprocessor` 上跑出的结果大约是百分之七十多。查看 wsdp 源码以及 `full_pipeline_xrf55.py`、`configurable_processor_xrf55.py` 找出根因，给出最小修复方案，先不要进行修改，先给我方案。
>
> 另外 `wsdp_pipeline_diagnostics_session_report.md` 是同事和别的大模型的对话，也可以参考。

### 助手回答/执行摘要

我先检查了 WSDP 源码、`full_pipeline_xrf55.py`、`configurable_processor_xrf55.py`，并参考了已有诊断文档。最初判断的方向包括：

- `full_pipeline_xrf55.py` 使用的可配置 pipeline 和源码 `BaseProcessor` 并不完全等价。
- `BaseProcessor` 的核心流程是：

```python
phase_calibration(whole_csi, dataset=dataset)
wavelet_denoise_csi(whole_csi)
```

- 可配置 pipeline 的执行顺序在 `execute_pipeline()` 里固定为：

```text
denoise -> outliers -> calibrate -> normalize -> interpolate -> ...
```

所以即使字典里写了 `calibrate` 在前，实际也可能先执行 `denoise`。

- 初期怀疑包括：
  - pipeline 顺序不一致；
  - `dataset` 没有传给算法；
  - `CSIDataset` 对复数数据默认 `np.abs()`；
  - z-score 后正负号被取绝对值折叠；
  - 数据划分方式和作者实验协议不一致。

---

## 二、关于 preset 不只是 fast 的问题

### 用户问题

用户指出：

> wsdp 中预设的算法组合不只是有 `apply_preset("fast")`，还有很多其他组合，对于其他组合 acc 仍然很低，看一下原因。

之后又补充：

> 这里每一个都有 z-score。

用户展示了可用 preset：

```text
high_quality:
  denoise: butterworth
  calibrate: stc
  normalize: z-score

fast:
  denoise: savgol
  calibrate: linear
  normalize: min-max

robust:
  denoise: wavelet
  calibrate: robust
  normalize: z-score

gesture_recognition:
  denoise: butterworth
  calibrate: stc
  normalize: z-score
  interpolate: cubic target_K=30

activity_detection:
  denoise: savgol
  calibrate: polynomial
  normalize: z-score

localization:
  denoise: wavelet
  calibrate: robust
  normalize: z-score
  interpolate: cubic target_K=64
```

### 助手回答/执行摘要

我解释过：

- 并不是所有 preset 都是 z-score，`fast` 设计上是 `min-max`。
- 但后来进一步检查源码发现一个重要问题：

```python
method = params.pop("method")
func = get_algorithm(category, method)
result = func(result, **params)
```

这会导致 `method` 被拿掉后没有传给算法函数。

所以：

```python
{"normalize": {"method": "min-max"}}
```

实际可能调用成：

```python
normalize_amplitude(result)
```

而 `normalize_amplitude()` 默认参数是：

```python
method="z-score"
```

也就是说：

```text
你以为 fast 用的是 min-max，
实际可能还是 z-score。
```

---

## 三、`execute_pipeline()` 报错：unexpected keyword argument dataset

### 用户问题

用户运行时报错：

```text
TypeError: execute_pipeline() got an unexpected keyword argument 'dataset'
```

报错位置：

```python
cleaned_csi = execute_pipeline(whole_csi, pipeline_steps, dataset=dataset)
```

用户要求：

> 解决一下报错。

### 助手回答/执行摘要

我检查了当前安装/本地源码的 `execute_pipeline()` 签名，发现它有的版本支持 `dataset`，有的版本不支持。

于是采用兼容写法：

```python
_EXECUTE_PIPELINE_ACCEPTS_DATASET = (
    "dataset" in inspect.signature(execute_pipeline).parameters
)

if _EXECUTE_PIPELINE_ACCEPTS_DATASET:
    cleaned_csi = execute_pipeline(whole_csi, pipeline_steps, dataset=dataset)
else:
    cleaned_csi = execute_pipeline(whole_csi, pipeline_steps)
```

目的：

- 老版本不传 `dataset`，避免报错；
- 新版本支持时传入 `dataset`，让算法可以拿到数据集信息。

---

## 四、查看 WSDP 版本与为什么最新版不支持

### 用户问题

用户问：

> 看一下我的 wsdp 的版本是多少，最新的版本是多少。

之后又问：

> 为什么最新的版本却不支持？

### 助手回答/执行摘要

我检查了环境中安装的 WSDP 和本地源码，说明了：

- 包版本和 GitHub 源码不一定完全一致。
- 即使是“最新版本”，也可能 API 没有暴露 `dataset` 参数。
- 文档、源码、pip 包之间可能存在版本差异。
- 当前项目里更稳的做法是：
  - 优先使用本地源码；
  - 在调用时检查函数签名；
  - 做兼容处理。

---

## 五、写 acc 低根因文档

### 用户问题

用户要求：

> 总结一下 acc 低的根本原因，详细一点，如何解决。写一个 md 文档放在当前文件夹。

### 助手回答/执行摘要

我创建了 `acc_low_root_cause_and_fix.md`，总结当时判断的低准确率原因，包括：

- `GroupShuffleSplit` 按用户划分导致训练用户和测试用户不同；
- 当前 XRF55 前三用户实验相当于用部分用户训练，再测试未见用户；
- `CSIDataset` 可能对 z-score 后的负数做 `np.abs()`，折叠正负号；
- 可配置 pipeline 与 `BaseProcessor` 顺序不一致；
- 源码中的算法组合不一定适合 XRF55 当前 `.npy` 数据；
- 最小修复建议包括：
  - 明确评估协议；
  - 做随机划分版本；
  - 保留 z-score 符号；
  - 本地脚本验证后再改源码。

---

## 六、为什么 val acc 也很低

### 用户问题

用户展示训练日志：

```text
Epoch [1/20] | Train Acc: 14.80% | Val Acc: 3.82%
Epoch [3/20] | Train Acc: 28.14% | Val Acc: 3.70%
Epoch [5/20] | Train Acc: 51.50% | Val Acc: 4.30%
```

问：

> 为什么这里的 val acc 都这么低，查找一下根因。

### 助手回答/执行摘要

我指出：

- 训练集 acc 能上升，说明模型和标签不是完全坏的。
- val acc 长期接近随机，说明 val 分布和 train 分布差异很大。
- 当时的主要怀疑是按用户划分导致：

```text
train users != val users
```

即训练用户和验证用户不同，模型学到的用户相关模式无法迁移。

---

## 七、写 `full_pipeline_xrf55_v2.py`：随机划分版本

### 用户问题

用户要求：

> 帮我再写一版 `full_pipeline_xrf55_v2.py`。你的意思是用户划分上出了问题，那么在这一版按照你说的来使用随机划分。其他部分不要变，和 `full_pipeline_xrf55.py` 一样。

### 助手回答/执行摘要

我创建了 `xrf55_test/full_pipeline_xrf55_v2.py`：

- 逻辑基本保持原版；
- 将数据划分改成普通随机划分；
- 保存位置仍按当时的输出逻辑。

后续用户运行后得到：

```text
Epoch [20/20] | Train Acc: 84.42% | Val Acc: 72.32%
测试集准确率: 0.6835
```

这说明：

```text
随机划分下结果正常很多。
```

---

## 八、解释 XRF55 的 55 个动作类别

### 用户问题

用户问：

> 55 个动作类别是什么意思，解释一下这个数据集。

### 助手回答/执行摘要

我解释：

- XRF55 是人体活动识别数据集；
- 文件名一般形如：

```text
01_20_08.npy
```

含义通常是：

```text
user_id = 01
action_id = 20
trial_id = 08
```

- 55 个动作类别就是 `action_id` 从 1 到 55；
- 每个用户每个动作有多次重复采集；
- 当前本地数据后来确认是：

```text
11 个用户
55 个动作
每个动作 20 次
总文件数 12100
```

---

## 九、ElderAL 是否也有划分问题

### 用户问题

用户问：

> 如果 XRF55 随机划分结果正常，再看一下 `full_pipeline_elderAL.py`。这个文件里面数据划分还是使用之前的划分逻辑，这个有问题吗，先不要修改我的代码。

### 助手回答/执行摘要

我检查了 ElderAL 的任务和 split 逻辑，说明：

- ElderAL 的任务和 XRF55 不同；
- ElderAL 是 activity/location 相关任务；
- 按 position 或其他 group 划分可能更符合任务；
- 不能简单把 XRF55 的结论套到 ElderAL；
- 但为了对照，也可以创建随机划分版本。

---

## 十、写 `full_pipeline_elderAL_v2.py`

### 用户问题

用户要求：

> 新写一版 `full_pipeline_elderAL_v2.py`，这个文件按照普通划分逻辑，其他不要变，文件放在 `elderAL_test` 文件夹下。

### 助手回答/执行摘要

我创建了 `elderAL_test/full_pipeline_elderAL_v2.py`，逻辑与原版相同，只把划分改成普通随机划分。

---

## 十一、修改多个 pipeline 的结果保存位置

### 用户问题

用户要求修改：

```text
full_pipeline_elderAL.py
full_pipeline_elderAL_v2.py
full_pipeline_xrf55_v2.py
full_pipeline_xrf55.py
```

保存逻辑：

```text
xrf55_test/result/robust+csimodel/best_checkpoint.pth
xrf55_test/result/robust+csimodel/loss_curve.png
```

### 助手回答/执行摘要

我修改了保存目录，使结果保存在各自测试文件夹下的 `result` 目录中。

后来用户指出：

> 你这样写不对，我之后会修改 `apply_preset("robust")` 的内容，你应该拿到 `apply_preset("robust")` 括号中的内容，然后加 model name 一起命名。

我进一步改成：

```python
preset_name = "robust"
pipeline_steps = apply_preset(preset_name)
model_name = "csimodel"
output_dir = os.path.join(..., "result", f"{preset_name}+{model_name}")
```

---

## 十二、robust 随机划分后仍然不高

### 用户问题

用户展示日志：

```text
Epoch [20/20] | Train Acc: 38.30% | Val Acc: 26.26%
测试集准确率: 0.2532
```

问：

> 这里我仍然使用预设的算法，比如 robust，但是修改了划分逻辑，可以看到 val acc 和 train acc 都有变化，但是不高，为什么？

### 助手回答/执行摘要

我说明：

- 随机划分解决的是评估协议问题；
- 但 `robust` 本身不一定适合当前 XRF55；
- `robust` 包含：

```text
wavelet
robust phase calibration
z-score
```

- 当前 XRF55 `.npy` 后来确认是实数幅度数据，相位校准基本没意义；
- z-score 后又可能被 `CSIDataset` 的 `np.abs()` 折叠；
- 因此 preset 本身可能会削弱信息。

---

## 十三、写 `detail_fixed.md` 并按文档修改源码

### 用户问题

用户要求：

> 按照 `acc_low_root_cause_and_fix.md` 你给出的修改方案，方案 B、C。我已经把源码拷贝到了当前文件夹下面，你看一下应该在哪里进行修改，把所有需要修改的部分再给出一个 `detail_fixed.md` 文档告诉我，详细告诉我在哪里修改，对应哪个文件。你先不要修改，先给我这个文件。

之后问：

> 方案 B 和 C 都能实现同样的效果吗还是承接顺序？

之后要求：

> 你按照 `detail_fixed.md` 文档来修改一下源码。

### 助手回答/执行摘要

我创建了 `detail_fixed.md`，说明需要改的源码文件：

- `src/wsdp/algorithms/registry.py`
  - `execute_pipeline()` 支持 `dataset`
  - 保留并传递 `method`
- `src/wsdp/processors/configurable_processor.py`
  - 调用 `execute_pipeline()` 时传入 `dataset`
- `src/wsdp/algorithms/denoising.py`
  - 对 real / 虚部为 0 的输入，返回 real
- `src/wsdp/algorithms/amplitude.py`
  - 对 real / 虚部为 0 的输入，返回 real
- `src/wsdp/datasets/CSIDataset.py`
  - 区分 real 和 complex，避免 real 数据无脑 `np.abs()`

之后我按文档修改了本地源码相应部分。

---

## 十四、写 v3 使用本地源码

### 用户问题

用户要求：

> 再写一个 `full_pipeline_elderAL_v3.py` 和 `full_pipeline_xrf55_v3.py`。这一次不调用 pip 包下载的 wsdp，import 使用源码的文件。其他逻辑和 v2 一致。

### 助手回答/执行摘要

我创建了：

```text
xrf55_test/full_pipeline_xrf55_v3.py
elderAL_test/full_pipeline_elderAL_v3.py
```

在脚本开头插入本地源码路径：

```python
LOCAL_WSDP_SRC = ...
sys.path.insert(0, LOCAL_WSDP_SRC)
```

确保使用当前工作区源码，而不是环境里安装的 pip 包。

---

## 十五、v3 后 XRF55 仍低，继续分析源码

### 用户问题

用户运行 v3 后展示：

```text
Epoch [20/20] | Train Acc: 40.42% | Val Acc: 24.58%
测试集准确率: 0.2444
```

问：

> 我按照 V3 跑出来的结果是这样的，acc 还是很低为什么，是什么原因，怎么改，先找一下根因，告诉我，先别改。

之后要求：

> 也就是说我们还要再修改一下源码是吧，详细说明一下需要再改哪里，然后更新一下 `detail_fixed.md`。更新完之后我先看一下然后你再进行修改。

之后要求：

> 现在你来修改一下源码相应部分。

### 助手回答/执行摘要

我进一步定位：

- 初次源码修改后仍低，说明只修 `dataset` 传参还不够。
- `wavelet_denoise_csi()` 对 real 输入会包装成 complex，虚部为 0。
- 后续 z-score 产生负数。
- 如果 dtype 是 complex，`CSIDataset` 看到 complex 后会 `np.abs()`，导致负数被折叠。

我更新 `detail_fixed.md`，并修改：

- `denoising.py`
- `amplitude.py`
- `CSIDataset.py`

使 real / 虚部约为 0 的数据不再错误走 complex 逻辑。

---

## 十六、检查 XRF55 v2 是否使用源码

### 用户问题

用户问：

> 我现在修改了一下 `xrf55_v2`，现在它使用的应该是源码，你看一下是吗？

### 助手回答/执行摘要

我检查 `sys.path` 插入顺序和 import 位置，确认：

- 如果脚本在导入 `wsdp` 前插入本地源码路径；
- 且路径指向 `SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/src`；
- 那么使用的是本地源码。

---

## 十七、源码修改后准确率仍不高

### 用户问题

用户展示：

```text
Epoch [20/20] | Train Acc: 57.76% | Val Acc: 31.52%
测试集准确率: 0.3098
```

问：

> 修改之后准确率还是不高，这次我使用的是 `xrf55_v2` 来跑的测试。

### 助手回答/执行摘要

我分析：

- 源码修复有提升，但仍不高；
- 说明 z-score 折叠等问题不是唯一根因；
- XRF55 的评估协议、用户泛化、训练用户数量仍是主要因素；
- 需要继续验证原版作者的 split 和当前实验是否一致。

---

## 十八、确认 robust preset

### 用户问题

用户问：

> robust 原来的不是这样的吗？

```text
robust
  denoise: wavelet
  calibrate: robust
  normalize: z-score
```

### 助手回答/执行摘要

我确认：

- 是的，`robust` preset 原始定义就是：

```python
"robust": {
    "denoise": {"method": "wavelet"},
    "calibrate": {"method": "robust"},
    "normalize": {"method": "z-score"},
}
```

并把这部分内容补充到 `detail_fixed.md`。

---

## 十九、查看 BaseProcessor 是什么组合

### 用户问题

用户问：

> 看一下源码中的 baseprocess 是什么算法处理组合。

### 助手回答/执行摘要

我查看 `base_processor.py`，说明：

```python
whole_csi = phase_calibration(whole_csi, dataset=dataset)
cleaned_csi = wavelet_denoise_csi(whole_csi)
```

所以 BaseProcessor 等价于：

```python
{
    "calibrate": {"method": "linear"},
    "denoise": {"method": "wavelet"},
}
```

但注意：

- BaseProcessor 是先 `calibrate` 再 `denoise`。
- `execute_pipeline()` 默认顺序是先 `denoise` 再 `calibrate`。

---

## 二十、保存训练过程到 `train_process.txt`

### 用户问题

用户要求修改多个 pipeline：

> 将模型训练过程写入对应文件夹下，命名为 `train_process.txt`。例如 `result/V3+robust+csitime/train_process.txt`。

涉及：

```text
full_pipeline_xrf55_v3.py
full_pipeline_xrf55_v2.py
full_pipeline_xrf55.py
full_pipeline_elderAL_v3.py
full_pipeline_elder_v2.py
full_pipeline_elder.py
```

### 助手回答/执行摘要

我给相关脚本添加 `Tee` 类：

```python
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
```

训练阶段：

```python
train_log_file = open(train_process_path, "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, train_log_file)
```

训练后恢复：

```python
sys.stdout = original_stdout
train_log_file.close()
```

---

## 二十一、写 XRF55 全 preset 全模型测试脚本

### 用户问题

用户给出模型列表和 preset 列表，要求：

> 在 `xrf55_test` 文件夹下面写一个脚本，测试所有可用预设和模型。比如 high_quality 下面测试所有模型，然后 fast 下面测试所有模型，以此类推。结果保存逻辑和 `full_pipeline_xrf55_v2.py` 一致。

### 助手回答/执行摘要

我创建了：

```text
xrf55_test/full_test_presets_models_xrf55.py
```

主要逻辑：

- 遍历 `list_presets()`
- 遍历 `list_models()`
- 每个组合保存到：

```text
xrf55_test/result/V2+preset+model/
```

- 保存：

```text
best_checkpoint.pth
loss_curve.png
train_process.txt
summary.csv
```

- 跳过大模型：

```text
visiontransformercsi
mambacsi
```

后续也加入跳过：

```text
graphneuralcsi
```

---

## 二十二、总结最开始个位数 acc、几次修复和提升

### 用户问题

用户要求：

> 现在你再总结一下最一开始我们 acc 个位数的原因，以及我们修复之后还是比较低的原因，接着我们又进行了一次修复，acc 有了提升。总结一下这几次修复。

### 助手回答/执行摘要

我总结：

1. 最开始个位数：
   - 按用户划分导致未见用户测试；
   - pipeline 与 BaseProcessor 不一致；
   - `dataset` 没有传递；
   - z-score 后正负号可能被 `np.abs()` 折叠。

2. 第一次修复后仍低：
   - 修了 `dataset`、real/complex 等问题，但跨用户泛化仍难；
   - preset 本身不一定适合 XRF55。

3. 再次修复后有提升：
   - 使用本地源码；
   - 改 Dataset 和算法处理 real 数据；
   - 一些模型/组合提升到 30%-50% 左右；
   - 但严格跨用户仍明显低于随机划分。

---

## 二十三、XRF55 是不是复数数据

### 用户问题

用户问：

> XRF55 数据集不是复数类型的 CSI 数据吗？顺便看一下 data 文件夹下面其他的数据集是不是。

之后又指出 README 表格里 XRF55 写的是复数。

### 助手回答/执行摘要

我检查本地数据：

```python
np.iscomplexobj(raw_data)
```

结果：

```text
data/xrf55/wifi/*.npy: float64, complex=False
```

说明当前本地 `.npy` 文件实际是实数幅度。

我解释：

- README 说 XRF55 支持/原始格式是 complex，可能对应原始 `.dat` 或理论数据；
- 当前本地 `data/xrf55/wifi/*.npy` 是已经处理成实数幅度的版本；
- 所以不能只看 README 表格，要以实际文件 dtype 为准。

---

## 二十四、写 fixed_doc.md / fided_doc.md

### 用户问题

用户要求：

> 列一下你修改过的源码里的文件，写一个 `fided_doc.md`。文件中只写你修改过哪些源码里的文件，修改了什么，为什么这样修改，目的是什么。不需要写太多，简洁一点即可。

之后问：

> 虚部几乎为零还是严格为零？

之后要求：

> 修改一下 `fixed_doc.md`，把主要修改的代码写上去。

之后问：

> 自始至终你有修改任何的具体算法的实现吗？

之后问：

> no-op 是什么意思？

### 助手回答/执行摘要

我创建并更新了：

```text
fided_doc.md
fixed_doc.md
```

说明修改文件：

- `registry.py`
- `configurable_processor.py`
- `denoising.py`
- `amplitude.py`
- `CSIDataset.py`

关于虚部：

- 对 XRF55 `.npy`，原始数据是严格实数；
- 某些算法会包装成 complex，虚部严格或接近 0；
- 判断时用阈值 `1e-10` 处理。

关于是否修改算法：

- 没有改变算法数学思想；
- 只是加了类型保护、参数传递和 real 数据处理分支；
- 对 real 输入避免错误走 complex/abs 逻辑。

no-op 解释：

```text
no-op = no operation
就是这个步骤实际什么也没做，直接返回原输入。
```

---

## 二十五、解释 `registry.py` 两段代码

### 用户问题

用户给出新旧 `execute_pipeline()` 代码截图，问：

> 这两段代码解释一下你修改的地方。

又问：

> 默认逻辑是什么样的？

### 助手回答/执行摘要

我解释：

旧逻辑：

```python
for category in order:
    if category in steps:
        params = steps[category].copy()
        method = params.pop("method")
        func = get_algorithm(category, method)
        result = func(result, **params)
```

问题：

- `method` 被 `pop` 掉；
- `normalize_amplitude()` 收不到 `method`;
- `dataset` 也传不进去；
- 算法只能走默认逻辑。

新逻辑：

```python
call_kwargs = params.copy()
signature = inspect.signature(func)

if "method" in signature.parameters:
    call_kwargs["method"] = method

if dataset is not None and "dataset" in signature.parameters:
    call_kwargs["dataset"] = dataset

result = func(result, **call_kwargs)
```

默认逻辑举例：

- `normalize_amplitude()` 默认 `method="z-score"`；
- `phase_calibration()` 如果没 dataset，就用默认 subcarrier indices；
- `interpolate_grid()` 如果没 dataset，也用默认频率位置。

---

## 二十六、解释 `configurable_processor` 传 dataset

### 用户问题

用户问：

> 解释一下修改原因，为什么：

```python
cleaned_csi = execute_pipeline(whole_csi, pipeline_steps, dataset=dataset)
```

### 助手回答/执行摘要

我解释：

- `dataset` 能让算法知道当前数据集；
- 对相位校准、插值这类算法，子载波位置可能依赖数据集；
- 不传 dataset 时只能走默认逻辑；
- 对 XRF55/Widar/Gait 这类 30 子载波数据，正确的 subcarrier mapping 很重要。

---

## 二十七、解释 `wavelet_denoise_csi()` 修改

### 用户问题

用户问：

> 解释一下这部分修改：

```python
input_is_effectively_real = np.isrealobj(csi_tensor)
...
if input_is_effectively_real:
    return denoised_amplitude
denoised_csi_tensor = denoised_amplitude * np.exp(1j * phase)
return denoised_csi_tensor
```

### 助手回答/执行摘要

我解释：

- 旧逻辑总是返回 complex：

```python
denoised_amplitude * exp(1j * phase)
```

- 对 XRF55 实数幅度来说，phase 实际是 0；
- 返回 complex 会让后续 Dataset 误以为是复数 CSI；
- 最终可能 `np.abs()` 折叠 z-score 正负；
- 新逻辑判断如果输入本质是 real，则直接返回 real denoised amplitude。

---

## 二十八、训练是否卡住、如何继续

### 用户问题

用户多次问：

> 检查一下现在是否还在训练，训练一轮耗时这么长吗？

> 再看一下训练是不是卡住了。

> 你估计多长时间跑完一个 epoch？

用户还提到：

> 我认为在运行上一个模型的时候显存不足，然后终止了运行，是不是会导致下面的模型卡住。

### 助手回答/执行摘要

我检查进程和训练日志，说明：

- 某些大模型如 `mambacsi`、`visiontransformercsi` 会占用显存较高；
- OOM 后如果进程没退出干净，可能残留显存；
- 建议：
  - 跳过大模型；
  - 每个模型结束后 `torch.cuda.empty_cache()`；
  - 必要时杀掉残留 Python 进程；
  - 用 summary 续跑，避免已完成组合重复跑。

后来用户要求：

> 对后续的 `visiontransformercsi`、`mambacsi` 在所有预设上跳过，同时 wider 也跳过。

我修改了 XRF55 和 Wider 的批量脚本跳过这两个模型。

---

## 二十九、写 Wider 测试代码

### 用户问题

用户要求：

> 在 `wider_test` 文件夹中写一下对 wider 数据集进行测试的代码，参考 xrf55 测试代码。仍然使用本地 wsdp 源码。代码保存逻辑和 xrf55 一致。需要写三个文件：
>
> - `configurable_processor_wider.py`
> - `full_pipeline_wider.py`
> - `full_test_presets_models_wider.py`
>
> 注意，这里的 gpu 我使用 cuda1 来跑。

### 助手回答/执行摘要

我创建了对应脚本：

```text
wider_test/configurable_processor_wider.py
wider_test/full_pipeline_wider.py
wider_test/full_test_presets_models_wider.py
```

特点：

- 使用本地 WSDP 源码；
- 保存到 `wider_test/result/...`；
- 使用 `cuda:1`；
- 批量测试时跳过 `visiontransformercsi`、`mambacsi`。

---

## 三十、写 Gait 测试代码

### 用户问题

用户要求：

> 在 `gait_test` 文件夹中写对 gait 数据集进行测试的代码，参考 xrf55。仍然使用本地 wsdp 源码。代码保存逻辑和 xrf55 一致。需要写三个文件：
>
> - `configurable_processor_gait.py`
> - `full_pipeline_gait.py`
> - `full_test_presets_models_gait.py`
>
> 注意，这里的 gpu 我使用 cuda0 来跑。
>
> 对于 `visiontransformercsi`、`mambacsi` 跳过。

### 助手回答/执行摘要

我创建了对应脚本，并将 Gait 的任务语义从 activity 改为 User ID：

```text
label = user_id
group = track_id * 100 + receiver_id
```

后续又根据实际测试目录切换到：

```text
test_gait/pipline_gait_v2.py
test_gait/full_test_presets_models_gait.py
```

并使用 `cuda:1` 进行批量测试。

---

## 三十一、Widar/Gait 上的 z-score 和复数问题

### 用户问题

用户问：

> 检查一下 xrf55 的 `V2_all_presets_models_summary.csv`。这些结果都没有特别高。看一下所有预设算法组合在 xrf55 上是否有问题，是不是有一些会出现和之前一样的问题，比如负数被折叠之类的。
>
> 同时也看一下 wider 和 gait，这些所有预设算法组合会不会也有其他问题。

之后用户要求：

> 用大白话告诉我这个应该怎么解决。

### 助手回答/执行摘要

我检查并解释：

- XRF55 当前 `.npy` 是 real amplitude；
- z-score 后负数可能被 `Dataset np.abs()` 折叠；
- Widar/Gait 是 complex CSI；
- 如果 `CSIDataset(use_phase=False)`，最终也只看幅度：

```python
data_list = np.abs(data_list)
```

所以 z-score 后复数幅度可能也会被折叠。

大白话解释：

```text
如果模型只看信号强弱，不看相位，
那就不要把 z-score 后的数据重新包装成 complex，
否则最后 Dataset 又会 abs，把负号变正。
```

解决方案分两种：

1. 幅度版：
   - complex -> amplitude
   - normalize
   - 输出 real
   - Dataset 保留正负号

2. 幅度 + 相位版：
   - complex -> amplitude + phase 双通道
   - 模型输入通道翻倍

用户提出：

> 既然是复数数据，也应该看一下相位。这样识别准确率应该会高，你认为这个解决方案怎么样？

我回答：

- 这个思路合理；
- 对 Widar/Gait 这种真正 complex CSI，相位可能有用；
- 但需要注意模型输入维度变化；
- 应先在自己的测试脚本里做 v1 幅度版、v2 幅度+相位版，再决定是否改源码。

---

## 三十二、写数据集问题总结文档

### 用户问题

用户要求：

> 总结一下这几个数据集出现的问题以及解决方案，写成一个 md 文档。

之后又要求：

> 你先修改一下文档里的表述。我的想法是，暂时先不要修改源码，我们在自己的代码上修改，写 V1 和 V2 版本，分别是先幅度版，再幅度+相位版。跑一下结果看看哪一个效果更好然后去改源码。一定采用最小解决方案，不要影响 elderAL 和 xrf55。

### 助手回答/执行摘要

我创建并修改了：

```text
dataset_pipeline_issue_solution_summary.md
```

内容包括：

- XRF55 当前本地 `.npy` 是 real amplitude；
- ElderAL 是 real CSV，不应受 complex phase 逻辑影响；
- Widar/Gait 是 complex CSI；
- 对 Widar/Gait 应先本地测试：
  - V1：幅度版；
  - V2：幅度+相位版；
- 不直接改源码，避免影响 XRF55/ElderAL。

---

## 三十三、写 Wider/Gait V1/V2

### 用户问题

用户要求：

> 现在你写 wider 和 gait 的 V1 和 V2 版本的代码。已经有的代码不要覆盖掉，重新写一个。

之后问：

> 说明一下哪些地方指明只用幅度，哪些地方指明用幅度加相位。

又问：

> 最一开始的 `configurable_processor_gait`、`wider` 版本是什么逻辑呢？

又确认：

> 所以最一开始也是只是用的幅度信息，但是正负号丢失了？

### 助手回答/执行摘要

我创建了 Wider/Gait 的 v1/v2 文件。

说明：

- V1 幅度版：
  - 使用 amplitude；
  - 输出 real；
  - 不带 phase。

- V2 幅度+相位版：
  - amplitude 和 phase 拼接；
  - 最后一维通道变多；
  - 模型同时看信号强弱和相位变化。

最开始版本逻辑：

```text
complex CSI -> CSIDataset -> np.abs()
```

也就是只用幅度。

如果前面 z-score 后再包装成 complex，负数还会被 `abs()` 折叠。

---

## 三十四、BaseProcessor 原始划分是否随机

### 用户问题

用户问：

> 看一下最一开始源码里的 baseprocessor 是随机划分吗？

之后给出 `create_data_split` 搜索结果，问：

> 看一下这里的文件，一开始的划分逻辑是什么样的。

### 助手回答/执行摘要

我检查了源码：

- `BaseProcessor` 只负责处理数据和返回：

```text
data, labels, groups
```

- 划分逻辑在 core / scripts 里；
- 通常使用 `GroupShuffleSplit`；
- 如果 group 数太少，才可能回退到普通随机划分；
- 对 XRF55，group 是 `user_id`；
- 所以最初不是样本级随机划分，而是按用户 group 划分。

---

## 三十五、为什么作者按用户划分效果可以，我这里不行

### 用户问题

用户问：

> 那为什么我在测试 xrf55 的时候按用户划分但是 val acc 上不去呢，源码作者是可以的，效果不错。

之后又给出一段代码文件，让分析：

> 这是怎么划分的数据。

之后要求：

> 分析一下改成随机划分后有没有数据泄露问题，然后把你跑的结果记下来。也分析一下为什么在 xrf55 上会跑出来 50% 的 acc。

### 助手回答/执行摘要

我创建了：

```text
xrf55_random_split_leakage_analysis.md
```

说明：

- 随机划分会让同一个用户、同一个动作、相近 trial 分布在 train/val/test；
- 这对评估跨用户泛化来说属于数据泄露；
- 但对“同用户/同分布 activity 识别”来说，是另一种协议；
- 随机划分能跑到 50%-70%，说明数据和模型不是坏的；
- 按用户划分低，是因为跨用户泛化困难。

后来用户强调：

> 三折留一用户我已经做过了，三折都低。是别的原因。XRF55 任务是 activity，和用户关系应该不是很大，源码作者跑出来更好。可能是算法或其他处理层面，再找根因。

我继续深入检查，并最终确认：

- XRF55 中用户特征非常强；
- 即使任务是 activity，信号里仍包含很强的人体/环境/采集域差异；
- 用前三用户做留一用户，训练用户太少。

---

## 三十六、只用前三用户跑 XRF55

### 用户问题

用户要求：

> 修改 `test_xrf55/full_test_presets_models_xrf55.py`，不要使用全部 xrf55 数据集，使用前三个用户的数据。可以参考 `pipline_xrf55.py`。

之后要求：

> 再修改 `full_test_presets_models_xrf55` 和 `pipline_xrf55` 使用 cuda0 跑代码。

### 助手回答/执行摘要

我修改：

```text
test_xrf55/full_test_presets_models_xrf55.py
test_xrf55/pipline_xrf55.py
```

使其：

- 只读取前三个用户；
- 使用：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_name = "cuda:0"
```

---

## 三十七、重新定位 XRF55 根因

### 用户问题

用户要求：

> 回顾所有之前对话，再次定位 xrf55 的 test acc 为什么这么低。我又重新使用 baseprocessor+CSIModel 训练了一下，结果在 `test_xrf55/result/baseprocessor+csimodel`。先来回顾一下 XRF55 为什么这么低的根因，这很重要。

之后指出：

> 三折留一用户我已经做过了，三折都低。是别的原因。再找一下根因，并且和用户泛化问题不大，因为 XRF55 的任务是 activity。和用户关系应该不是很大，并且 sdp 项目源码作者跑出来的结果比我的要好。可能是在算法或者其他处理层面。

### 助手回答/执行摘要

我进一步排查：

- `pipline_xrf55_v2.py` 验证合并 3 个 Rx 后仍低；
- 说明不是 XrfReader 拆样本导致；
- 检查本地 XRF55 `.npy` 是 real amplitude；
- 检查 BaseProcessor 的 phase calibration 在当前数据上是 no-op；
- 检查 `CSIDataset` 和 `execute_pipeline` 问题；
- 但 v3 修复后仍低；
- 最后通过小样本统计诊断确认用户特征非常强。

---

## 三十八、写 `pipline_xrf55_v2.py`

### 用户问题

用户要求：

> 暂时不修改源码，新写一个 `pipline_xrf55_v2.py`，这一版不把一个完整样本拆成 3 个残缺样本。一个 `.npy` 文件 -> 一个 `CSIData` 样本。不要求改源码。在文件最上方写上注释，以便最后整合。接下来我会用这一版测试。

之后用户补充：

> 不用直接调用源码，继续沿用 `pipline_xrf55.py` 的逻辑即可。源码里面把顺序定死的，其实 baseprocessor 和 BaseProcessor() 没区别。

### 助手回答/执行摘要

我创建：

```text
test_xrf55/pipline_xrf55_v2.py
```

关键注释：

```text
旧逻辑：一个 .npy -> 3 个 CSIData，每个 Rx 一个
新逻辑：一个 .npy -> 1 个 CSIData，3 个 Rx 和 3 根天线合并为 (30,9)
```

用户跑完后：

```text
Train Acc: 92.55%
Val Acc: 约 5%
Test Acc: 5.64%
```

用户指出：

> 所以并不是 XrfReader 把一个完整样本拆成 3 个的问题。

我确认：

```text
reader 拆样本不是主因。
```

---

## 三十九、限制诊断只跑前三个用户

### 用户问题

用户说：

> 只跑前三个用户诊断即可，我训练的时候也用的前三个用户。后面不需要。

### 助手回答/执行摘要

我将诊断范围限制为前三用户，得到关键结果：

```text
random_3users acc: 0.6687
within_user_1 acc: 0.8727
within_user_2 acc: 0.8970
within_user_3 acc: 0.8758
leave_user_1 acc: 0.0864
leave_user_2 acc: 0.0873
leave_user_3 acc: 0.1055
```

---

## 四十、更新 `baseprocessor_csimodel_root_cause_analysis.md`

### 用户问题

用户要求：

> 把你找到的根因更新到 `baseprocessor_csimodel_root_cause_analysis.md` 中，精简一点，用大白话说。

之后又问：

> 文档中说已确认沿用 gait v2 划分逻辑，gait v2 划分逻辑是什么，我忘记了。
>
> Widar sanity check 是什么？
>
> 把 wider 这一段用大白话说明一下。

之后指出：

> 文档没有变啊，你好像没有进行修改，现在修改一下。

### 助手回答/执行摘要

我修改了：

```text
baseprocessor_csimodel_root_cause_analysis.md
```

补充：

### Gait v2 划分逻辑

```text
label = user_id
group = track_id * 100 + receiver_id
```

大白话：

```text
模型要识别“是谁”；
划分时不按人切，而按走路轨迹 + 接收器切。
```

原因：

```text
如果把某个人整个放测试集，训练时模型从没见过这个人的标签，普通分类模型不可能预测这个人。
```

### Widar sanity check

我改成“大白话版基础检查”：

```text
先别急着测跨用户。
先确认数据有没有读错、标签有没有错、预处理有没有坏、模型能不能学。
```

`test_wider/pipline_wider_v2.py` 逻辑：

```text
label = gesture_type
group = position_id * 1000 + orientation_id * 100 + receiver_id
```

不是按用户分，而是按采集条件分。

---

## 四十一、解释 Wider v2 的划分逻辑

### 用户问题

用户选中：

```text
test_wider/pipline_wider_v2.py 已完成 Widar 基础检查...
```

问：

> 数据划分逻辑是什么样的？

之后要求：

> 用大白话说。

### 助手回答

我回答：

`pipline_wider_v2.py` 的划分逻辑是：

```text
label = gesture_type
group = position_id * 1000 + orientation_id * 100 + receiver_id
```

大白话：

```text
不是按人分，
而是按拍摄/采集条件分。
```

比如：

```text
某个位置 + 某个朝向 + 某个接收器
```

这一组数据只会放到训练集、验证集、测试集中的一个地方，不会混着放。

但同一个人，比如 `user2`，可能训练集里有，验证集里也有，测试集里也有。

所以这个实验不是问：

```text
模型能不能识别一个从没见过的新用户？
```

而是问：

```text
同一批用户都见过的情况下，换个位置、换个朝向、换个接收器，模型还能不能识别手势？
```

结果能到 `68.60%`，说明：

```text
数据大概率没读错
标签大概率没错
模型输入大概率没错
模型确实能学到手势
```

但它不能证明模型能跨用户泛化。跨用户要看 v3。

---

## 四十二、修改 `full_test_presets_models_gait.py` 续跑逻辑

### 用户问题

用户指出：

> 其实我们已经跑完了 high_quality 这个预设，除了跳过的模型以外。那么就没有必要再进行 high_quality 的处理了，看一下目前是哪个预设还没跑完，处理这个预设就好了。

### 助手回答/执行摘要

我检查：

```text
high_quality: 19/19 已记录
fast: 2/19 已记录
robust: 0/19
gesture_recognition: 0/19
activity_detection: 0/19
localization: 0/19
```

然后修改：

```text
test_gait/full_test_presets_models_gait.py
```

新增逻辑：

- 如果某个 preset 的所有模型组合都已记录，直接跳过这个 preset；
- 不再重新做 step 2 数据处理；
- 如果 preset 只跑了一部分，则只跑未记录模型；
- 配置里跳过的模型直接写入 summary。

---

## 四十三、再次从算法和源码角度分析 XRF55，并推荐组合

### 用户问题

用户要求：

> 这样你从算法的层面以及源码数据处理的角度找一下 xrf55 数据集的问题。或者你看一下所有可用的算法组合，可以自己组合。看看哪些算法适合处理 xrf55 的数据。这是两个任务：
>
> 1. 首先从算法的层面以及源码数据处理的角度找一下 xrf55 数据集的问题；
> 2. 第二个任务给出一个你认为好的算法组合。

### 助手回答

我检查源码和样本后回答：

1. 本地 XRF55 `.npy` 是实数幅度：

```text
raw dtype: float64
complex? False
```

2. 相位校准对当前 `.npy` 基本没意义。

3. `execute_pipeline()` 会丢掉 `method`，导致 `min-max` 可能实际变成默认 `z-score`。

4. `CSIDataset` 默认 `np.abs()`，会折叠 z-score 正负号。

我推荐两个组合：

```python
xrf55_amp_minmax = {
    "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
    "normalize": {"method": "min-max"},
}
```

和：

```python
xrf55_amp_robust_minmax = {
    "denoise": {"method": "butterworth", "order": 4, "cutoff": 0.25},
    "normalize": {"method": "min-max"},
}
```

但我也提醒：

```text
如果不修 execute_pipeline，这两个组合不一定真的跑到 min-max。
```

---

## 四十四、写 `pipline_xrf55_v3.py`

### 用户问题

用户说：

> 你推荐的这两个算法组合我都试了，效果不好。你先看一下。
>
> 看完之后尝试一下任务 1 你给出来的 XRF55 的源码和算法问题。仍然还是先不修改源码，你写一下 `pipline_xrf55_v3.py`。代码最上面写一下注释，源码和算法的问题，写详细问题是什么，这一版是求其进行修改，暂时不修改源码。写完 `pipline_xrf55_v3.py` 之后我们再对 xrf55 进行测试。
>
> 然后你再更新一下 `baseprocessor_csimodel_root_cause_analysis.md`。

### 助手回答/执行摘要

我创建：

```text
test_xrf55/pipline_xrf55_v3.py
```

文件顶部写明要验证的问题：

1. XRF55 `.npy` 是 real-valued amplitude；
2. phase calibration 对它基本无意义；
3. 源码 `execute_pipeline()` 会丢掉 `method`；
4. 源码 `CSIDataset` 会默认 `np.abs()`；
5. v3 不改源码，只在脚本里本地修正：
   - `local_execute_pipeline_fixed()`
   - `XRF55RealDataset`

v3 默认 pipeline：

```python
pipeline_steps = {
    "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
    "normalize": {"method": "z-score"},
}
```

用户后来修改/运行了 `v3_baseprocessor` 和 `v3_savgol_zscore_keep_sign`。

我也更新了：

```text
baseprocessor_csimodel_root_cause_analysis.md
```

---

## 四十五、v3 跑完后仍差，逐步检查

### 用户问题

用户说：

> 我使用 `pipline_xrf55_v3.py` 跑了代码，acc 还是很差。你再仔细看一下到底是哪里出了问题。详细检查一下每个步骤。你可以拿一小部分样本，按照你的流程来跑一下，具体看一下到底哪里有问题。但是不要写新的 py 文件，不要修改别的代码。仔细检查仔细检查。

### 助手回答/执行摘要

我没有写新文件，也没有修改代码，只用临时命令做诊断。

检查到的 v3 结果：

```text
v3_savgol_zscore_keep_sign:
Train Acc: 46.03%
Val Acc: 7.21%
Test Acc: 7.27%

v3_baseprocessor:
Train Acc: 74.02%
Val Acc: 7.39%
Test Acc: 5.58%
```

说明：

- 保留 z-score 正负号没有救回来；
- 修 Dataset abs 也不是核心根因；
- `savgol + zscore` 训练集学得更差；
- baseprocessor 训练能学，但跨用户 val/test 仍低。

然后我做文件级 split 复现：

当前前三用户：

```text
train users: [2, 3]
val users: [1]
test users: [1]
```

旧版 7518483 全 11 用户，seed=42：

```text
train users: [2, 4, 5, 7, 8, 9]
val users: [3, 11]
test users: [1, 6, 10]
```

关键差异：

```text
你现在只用 2 个用户训练；
旧版用 6 个用户训练。
```

然后做小样本统计特征诊断：

```text
predict user_id random acc: 94.34%
predict action_id mixed users random acc: 73.54%
predict action train user2/user3 -> user1 acc: 11.45%
```

再试简单归一化：

```text
raw:                 cross-user 11.45%
sample_zscore:       cross-user 7.64%
time_channel_zscore: cross-user 8.55%
log_sample_zscore:   cross-user 7.45%
```

最终结论：

```text
XRF55 数据里用户特征非常强。
同用户或随机混合用户时动作很好学。
只用 user2/user3 训练，去测 user1，动作泛化非常差。
```

我给出的根因优先级：

```text
第一根因：只用前三用户做跨用户测试，训练用户太少。
第二根因：XRF55 用户域差异很强，动作模式跨人不稳定。
第三根因：相位算法对当前 .npy 没用，但不是致命问题。
第四根因：Dataset abs / execute_pipeline method 是源码问题，但 v3 证明不是这次个位数 acc 的主因。
```

---

## 四十六、当前用户要求导出所有对话

### 用户问题

用户要求：

> 把当前对话框我和你到目前为止所有的对话输出为一个新的 md 文档，注意是所有的对话内容，我的问题，你的回答都必须包括。

### 助手回答/执行摘要

我说明：

```text
由于对话很长，中途发生过上下文压缩，压缩前的逐字助手回复不可恢复。
我会按当前可恢复内容导出：用户问题尽量完整，助手回答按可见原文和压缩摘要整理。
```

本文件即为本次导出。

---

## 附录 A：本轮工作区中确认创建/修改过的重要文档

```text
acc_low_root_cause_and_fix.md
detail_fixed.md
fided_doc.md
fixed_doc.md
dataset_pipeline_issue_solution_summary.md
xrf55_random_split_leakage_analysis.md
baseprocessor_csimodel_root_cause_analysis.md
conversation_export.md
```

其中当前仍存在并重点使用的是：

```text
baseprocessor_csimodel_root_cause_analysis.md
conversation_export.md
```

---

## 附录 B：本轮工作区中确认创建/修改过的重要脚本

### XRF55

```text
test_xrf55/pipline_xrf55.py
test_xrf55/pipline_xrf55_v2.py
test_xrf55/pipline_xrf55_v3.py
test_xrf55/full_test_presets_models_xrf55.py
test_xrf55/configurable_processor_xrf55.py
```

### Wider / Widar

```text
test_wider/pipline_wider.py
test_wider/pipline_wider_v2.py
test_wider/pipline_wider_v3.py
test_wider/pipline_wider_v4.py
test_wider/diagnose_wider_v4.py
test_wider/full_test_presets_models_wider.py
test_wider/configurable_processor_wider.py
```

### Gait

```text
test_gait/pipline_gait_v2.py
test_gait/full_test_presets_models_gait.py
```

### ElderAL

```text
test_elderAL
elderAL_test/full_pipeline_elderAL_v2.py
elderAL_test/full_pipeline_elderAL_v3.py
```

注：部分较早路径来自压缩摘要，当前工作区里可能已经被整理到 `test_*` 目录。

---

## 附录 C：关键结论总表

| 数据集 | 当前任务 | 主要问题 | 当前结论 |
| --- | --- | --- | --- |
| XRF55 | Activity | 前三用户跨用户测试极难；训练用户只有 2 个；用户域差异强 | 当前个位数 acc 主要不是源码单点 bug，而是评估设置过难 |
| Widar/Wider | Gesture | 3 用户设置下跨用户泛化弱，尤其 user2 | v2 condition split 能学，v3 LOO 跨用户下降 |
| Gait | User ID | 原始标签/划分语义错误 | v2 改为 `label=user_id`，按非用户 group 划分 |
| ElderAL | Act-Loc / Activity-Location | 当前不是异常重点 | 表现相对正常 |

---

## 附录 D：XRF55 最终阶段关键诊断结果

### 当前前三用户 split

```text
train: user2, user3
val:   user1
test:  user1
```

### 旧版 7518483 全量 11 用户 split

```text
train: user2, user4, user5, user7, user8, user9
val:   user3, user11
test:  user1, user6, user10
```

### 小样本统计特征诊断

```text
预测 user_id 随机划分: 94.34%
预测 action_id 混合用户随机划分: 73.54%
训练 user2/user3，测试 user1 的 action: 11.45%
```

### 简单归一化对跨用户的影响

```text
raw:                 cross-user 11.45%
sample_zscore:       cross-user 7.64%
time_channel_zscore: cross-user 8.55%
log_sample_zscore:   cross-user 7.45%
```

结论：

```text
简单滤波/归一化不能解决前三用户 leave-one-user 的低 acc。
当前最大问题是训练用户太少 + 用户域差异强。
```
