# CSI 预处理实验交接文档（2026-07-13）

## 1. 这项工作在做什么

目标是判断 CSI 的预处理算法是否真的有用，而不是只从大量组合中挑一个最高
`test acc`。当前在比较四类数据集上的原始/最小处理、预设处理和全量组合处理：

- **ElderAL**：全量数据，动作识别；模型主要为 `csitime`。
- **XRF55**：只用前三个用户；模型主要为 `resnet1d`。
- **Widar**：条件划分，新增了相位信息；预设测试和 320 个全量组合都在做。
- **Gait**：用户 ID 划分，新增了相位信息；全预设 x 全模型测试正在做。

核心问题是：某个处理链相对于 `raw-minimal`、现有 BaseProcessor 或预设，是否在
**同一批数据切分**下稳定提升准确率；若只在一个 seed 或一个 test set 上最高，不能
称为“稳定最佳”。

## 2. 重要术语和比较口径

### raw-minimal 不是完全零处理

`raw-minimal` 保留模型不可缺少的输入适配：读取、帧排序、丢弃无效样本、固定时间
长度的截断/补零、标签/group 解析，以及数据集读取器固有的幅度转换。

它**不做**去噪、异常值处理、相位校准、归一化和子载波插值。因此它是“原始 CSI
加最小输入适配”的基线，不等于把文件字节直接喂给模型。

### 三类实验

1. **raw-minimal**：判断模型不用额外信号处理时能达到什么水平。
2. **预设测试**：比较项目定义的预设方案和多种模型。
3. **80/320 全量组合**：固定模型，穷举/半穷举去噪、异常值、校准、归一化、插值。

不能把不同输入尺寸、不同模型容量、不同数据切分的分数当成纯算法效果。

## 3. 已完成的工作与结论

### 3.1 ElderAL：80 组和 raw-minimal 已完成

相关文件：

- 80 组：[elderAL_80_pipeline_csitime_summary.csv](test_elderAL/result/full_tests/elderAL_80_pipeline_csitime_summary.csv)
- raw 脚本：[full_test_raw_minimal_elderAL.py](test_elderAL/full_test_raw_minimal_elderAL.py)
- raw 输出目录：[raw_minimal_tests](test_elderAL/result/raw_minimal_tests)

已得到的结果：

- 80 组单次最高为 `savgol_w7_p3 + iqr + min-max + linear64`，`test acc=92.59%`；
  `elderAL_040` 也达到 `92.59%`。
- 最佳预设为 `high_quality`，`test acc=91.48%`。其配置是 Butterworth + STC（对
  当前 ElderAL 实数输入基本不起作用）+ z-score。
- raw-minimal 五次手动 seed 结果为 `87.04% / 73.33% / 77.90% / 91.39% / 100.00%`，
  均值 `85.93% +/- 10.63pp`。
- Hampel 最差：16 组均值约 `64.19%`。它在 11 帧局部窗口中把偏离局部中位数超过
  3 倍 MAD 的点换为中位数，可能把动作的快速峰值也当成异常点抹掉。
- 对本实现，`min-max` 普遍比 `z-score` 更好。一个重要实现原因是 ElderAL 的
  z-score 会给出正负值，后续 `CSIDataset` 又取绝对值，正负方向信息被折叠。
- `IQR + z-score`、特别是再接 bandpass/插值时存在明显差组合；不是 IQR 本身在所有
  情况下都无效，而是 IQR 削弱强动作点后，z-score 的符号折叠和插值失真会叠加。

**不能过度解读的地方**：ElderAL 有 9 个 position group。当前 `_create_data_split`
先抽取约 30% group，再把其中一部分作为验证集，最终 test 实际只有一个 position，
大约 11% 样本。五个 seed 对应的测试位置不同（已观察到 8/8/9/4/5），所以 raw 的
大波动主要混入了位置难度，不能用它证明某方案稳定好或坏。

### 3.2 XRF55（前三用户）：80 组和 raw-minimal 已完成

相关文件：

- 80 组：[xrf55_80_pipeline_resnet1d_summary.csv](test_xrf55/result/full_tests/xrf55_80_pipeline_resnet1d_summary.csv)
- raw 脚本：[full_test_raw_minimal_xrf55.py](test_xrf55/full_test_raw_minimal_xrf55.py)
- raw 输出目录：[raw_minimal_tests](test_xrf55/result/raw_minimal_tests)

已得到的结果：

- 80 组单次最高为 `savgol_w7_p3 + iqr + z-score + cubic15`，`test acc=85.15%`。
- raw-minimal 五个 seed 为 `83.48% / 85.00% / 83.03% / 81.52% / 83.33%`，均值
  `83.27% +/- 1.24pp`。
- 因而最佳全量方案相对 raw 的 seed=42 只高约 `1.67pp`，而 raw 的另一次 seed 已有
  `85.00%`。现有证据不足以证明该全量方案稳定优于 raw-minimal。
- bandpass 16 组均值约 `40.09%`，远差于其他去噪方法。主要风险是代码硬编码
  `fs=1000`、频段 `0.5-50Hz`，但 XRF55 的真实帧率没有被读取确认；频率含义可能整体
  错位，并且滤波可能去掉了分类需要的慢变化/整体形状。
- 当前使用的 XRF55 `.npy` 输入经检查是 **real `float64`**，不是复数 CSI。不要再用
  “分别滤实部和虚部”的解释来说明这批 XRF55 结果；那只适用于 reader 的另一条 `.dat`
  复数读取分支。

### 3.3 Widar：相位预设测试已分析完成

相关文件：

- 新相位预设：[widar_all_presets_models_summary.csv](test_wider/result/new_preset_tests/widar_all_presets_models_summary.csv)
- 旧版（未使用相位）：[condition_v2_all_presets_models_summary.csv](test_wider/result/preset_tests/condition_v2_all_presets_models_summary.csv)
- 早期汇总报告：[analysis_elder_xrf55_widar_phase.md](test_wider/result/analysis_elder_xrf55_widar_phase.md)

已完成的配对分析（84 个同名预设 x 模型组合）：

- 加入相位后平均 `+2.72pp`，中位数 `+2.22pp`。
- `61` 个组合提升，`23` 个下降；下降占 `27.38%`，平均下降 `1.84pp`。
- 新版最佳为 `activity_detection + mlpmodel`，`74.55%`；旧版全局最佳为
  `gesture_recognition + mlpmodel`，`68.69%`。最高分提高 `5.86pp`。
- 相位对 `resnet2d`、`bilstm`、`mlp` 的平均增益更明显；对 `fewsense`、
  `attentiongru`、`lstm` 有小幅平均下降。说明相位不是对每一个模型必然有益。

### 3.4 Gait：相位预设测试仍在进行

相关文件：

- 运行脚本：[full_test_presets_models_gait.py](test_gait/full_test_presets_models_gait.py)
- 新结果：[gait_all_presets_models_summary.csv](test_gait/result/new_preset_tests/gait_all_presets_models_summary.csv)
- 旧结果：[user_id_v2_all_presets_models_summary.csv](test_gait/result/preset_tests/user_id_v2_all_presets_models_summary.csv)

截至 2026-07-13 检查时：CSV 有 47 条，`44 ok / 3 failed`，尚未完成。

- 当前最高：`fast + mlpmodel`，`test acc=96.25%`。
- 之前已做的 32 个新旧可配对组合中，相位平均约 `+9.32pp`，中位数 `+7.90pp`；
  27 个提升、1 个持平、4 个下降。
- 此结论是阶段性结论，因为目前只跑完部分预设/模型，不能宣布 Gait 的最终最佳方案。

### 3.5 Widar：320 组全量组合正在运行

相关文件：

- 运行脚本：[full_test_widar_new.py](test_wider/full_test_widar_new.py)
- 组合顺序说明：[widar_optimized_320_pipeline_order.md](test_wider/widar_optimized_320_pipeline_order.md)
- 当前汇总：[widar_320_pipeline_optimized_mlpmodel_summary.csv](test_wider/result/full_tests_new/widar_320_pipeline_optimized_mlpmodel_summary.csv)

脚本固定 `mlpmodel`，组合空间为：去噪 5 x 异常值 2 x 相位校准 4 x 归一化 2 x
插值 4 = **320 组**。固定 seed=42、固定 Widar split，输出为幅度+相位实数通道。

截至 2026-07-13 检查时：已写入 **56/320（17.5%）**，全部 `ok`，且目前刚好都属于
`wavelet` 去噪分支，不能比较不同去噪方法。

- 当前最高：`widar_010`，`wavelet + iqr + polynomial_d3 + z-score + cubic15`，
  `test acc=71.52%`。
- 当前最低：`wavelet + iqr + robust + min-max + decimate15`，`51.92%`。
- 已跑完的 wavelet + IQR 块中，`linear`、`polynomial_d3`、`stc` 相位校准相近，
  `robust` 明显较差，尤其搭配 `min-max`。合理假设是 robust 去除了部分与动作有关的
  相位变化；但必须等后续去噪分支跑完才能确认是否普遍存在。
- 目前 `z-score` 在 Widar 全量 wavelet 分支优于 `min-max`，与 ElderAL 相反，再次说明
  不能把某种归一化当作通用赢家。

当前的 71.52% 低于 Widar 相位预设最佳 74.55%，但两者管线细节不同，且全量只完成
17.5%，不能据此判定全量组合不如预设。

## 4. 当前卡在哪里

不是代码无法运行，而是**实验还没跑完，且现有比较设计不足以证明稳定收益**：

1. Widar 320 组和 Gait 新预设仍在产出结果；不要在已有同一脚本运行时启动第二份，
   以免抢 GPU、覆盖日志或造成结果混乱。
2. ElderAL 的 group split 让不同 seed 评估的是不同 position，raw 五次结果不能直接
   用来估计随机训练波动。
3. raw-minimal 脚本的汇总 CSV 用 `"w"` 写入，会被每次改 seed 的运行覆盖：
   ElderAL 在脚本第 253 行，XRF55 在第 277 行。完整五次结果目前只能从各自 seed
   目录的 `train_process.txt` 恢复。
4. 80 组、预设和 raw 的子载波数/模型参数量并非都一致。例如 ElderAL full 将
   512 压到 64，模型参数量也从约 635,654 变为约 119,558。因此“全处理更高”不等于
   单独某个去噪或归一化算法有效。
5. 从 80 或 320 个组合中按 test acc 选冠军，会有多重挑选的乐观偏差。冠军必须在未
   用来挑选它的切分上复验。

## 5. 下一步建议（按优先级）

### A. 先让当前实验自然完成并做快照

1. 等 Widar 320 和 Gait 预设脚本完成；每次分析前记录 CSV 行数、`ok/failed` 数。
2. 对所有 `failed` 组合读取各目录的 `error.txt`，区分 CUDA OOM、数据形状问题和代码
   bug。预设测试中的 `cnn2d` 曾出现 OOM，不能把失败当成低准确率。
3. 完成后分别按：总排名、每个算法因子的平均/中位数、同后缀的成对差异，整理结果。

### B. 做真正能判断“处理是否有意义”的复验

在**相同切分、相同模型结构、相同输入尺寸**下，至少比较：

1. raw-minimal；
2. 只做子载波插值的基线（ElderAL -> 64，XRF55/Widar -> 15）；
3. 当前最佳预设；
4. 全量组合 top-3；
5. 可选：BaseProcessor 默认处理链。

每项用同一组 5 个 seed 跑完后，报告平均值、标准差、相对 raw 的逐 seed 差值、提升
次数。不要只报告最高 `test acc`。

ElderAL 更推荐做 **9 个 position 的 leave-one-position-out**：每次固定一个 position
作 test，其他位置训练/验证。最后报告 9 个 position 的平均和方差，才能回答算法是否
泛化到新位置。

### C. 在开始下一轮前的小代码修正

1. 把两个 raw-minimal 脚本的 summary 写入由 `"w"` 改为追加，或在文件名中包含 seed；
   防止覆盖历史记录。
2. 在 summary 中额外写入：seed、训练/验证/测试样本数、测试 group/position、输入
   shape、模型参数量、实际算法参数。
3. 对 bandpass，优先查明真实采样率；没有真实 `fs` 时，不应把 `fs=1000` 的结果解释为
   bandpass 算法本身无效。

## 6. 运行与接手注意事项

- 先检查 GPU 和正在运行的进程，再决定是否续跑。公共服务器上不能仅凭 GPU 占用断言
  某个 Python 进程属于当前用户。
- `full_test_widar_new.py` 和 `full_test_presets_models_gait.py` 都会从已有 summary 跳过
  已记录的组合，适合断点续跑；但应先确认该脚本没有另一实例运行。
- 所有相对路径以上述 `SDP/` 目录为基准；数据集位于项目同级的 `sdp_dataset`。
- 当前工作区存在用户实验产生的未跟踪结果和修改中的训练日志/权重。不要用
  `git reset --hard`、`git checkout --` 或清理命令去恢复它们。

## 7. 一句话结论

raw-minimal 已经是强基线，说明模型本身能从原始 CSI 学到不少信息；预处理并非越多越
好，且高度依赖数据集。现在有“部分链条有正向信号”的证据，但尚无任一方案能被称为
跨 seed、跨 position、稳定最佳。下一阶段的重点不是继续挑更高单次分数，而是做同口径
复验，量化预处理相对 raw 的稳定增益。
