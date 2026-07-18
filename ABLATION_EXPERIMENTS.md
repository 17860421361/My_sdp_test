# 消融实验运行说明

这些脚本全部位于仓库最外层，只调用现有 WSDP 源码接口，不修改现有源码、测试文件或 `.gitignore`。默认结果目录为：

```text
result/ablations/
```

建议从仓库根目录运行。正式结论优先看 3 个模型随机种子的均值和标准差，不要只看一次训练。

## 1. Robust 内部是哪一步造成下降

```bash
python ablate_robust_components.py --dataset gait
python ablate_robust_components.py --dataset widar
```

固定后续步骤为 `nearest15 + z-score + MLP`，避免复数混合干扰，分别比较：

- 不校准、linear 参考；
- 只减公共中位相位；
- 只做每子载波去趋势；
- 前 50 帧估计斜率；
- 全动作范围估计斜率；
- 完整 Robust。

判断方法：

- `common_only` 明显下降：支持“公共相位中包含动作信息”；
- `detrend_first50_only` 明显下降：支持“线性动作趋势被删除”；
- `first50` 差而 `all_frames` 恢复，同时后 1/4 相位修正远大于前 1/4：支持“前 50 帧外推造成累积畸变”；
- 完整 Robust 比两个单独步骤都差：说明两种损伤存在叠加。

只生成信号诊断、不训练：

```bash
python ablate_robust_components.py --dataset gait --skip-training
```

## 2. Robust 后为什么四种插值差距很大

```bash
python ablate_robust_interpolation.py --dataset gait
```

脚本包含：

- Robust 与 linear 相位校准下的 nearest、linear、cubic、decimate 配对；
- 不插值 30 子载波；
- 幅度和展开相位分别插值的 polar 对照；
- 不混合、直接选择 15 个原始子载波的 stride 对照；
- `|复数插值结果| / 插值(|复数|)` 抵消比例；
- Robust 前后相邻子载波相位跳变。

判断方法：

- linear 校准下四种方法接近、Robust 下才分开：证明是 `Robust × 插值` 交互，不是插值方法本身普遍较差；
- Cartesian 的抵消比例明显小于 1，且 polar 版本准确率恢复：支持“复数向量相消”；
- decimate 仍显著差于 polar、nearest 或 direct-stride：说明除复数相消外，还存在频域低通和非均匀子载波位置被当成均匀序列的问题；
- linear 与 cubic 的多 seed 置信区间重叠时，不能宣称 cubic 必然更差。

只生成信号诊断、不训练：

```bash
python ablate_robust_interpolation.py --dataset gait --skip-training
```

## 3. Robust 与 min-max 的额外交互

```bash
python ablate_robust_normalization_2x2.py --dataset widar
python ablate_robust_normalization_2x2.py --dataset gait
```

这是干净的 `2 × 2` 实验：

```text
相位校准：linear / robust
幅度归一化：z-score / min-max
```

四组统一使用：

```text
校准 -> nearest15 -> 归一化 -> 显式[幅度, 相位]
```

因此不会混入当前全量脚本中两种归一化执行顺序不同的问题。重点查看：

```text
result/ablations/robust_normalization_2x2/<dataset>/factorial_effects.json
```

其中交互项为负，表示 min-max 使 Robust 的损失进一步扩大。输入幅相标准差也会写入结果，用于检查 `[0,1]` 幅度与 `[-π,π]` 相位的尺度失衡。

## 4. ElderAL：Hampel 是否真的替换了动作信息

先运行不需要训练的直接诊断：

```bash
python elder_hampel_replacement_diagnostic.py
```

它会统计 window 1、2、3、5 下的：

- 标量、帧和样本替换率；
- `MAD=0` 比例；
- 零阈值替换占全部替换的比例；
- 时间总变化量和峰值保留率。

然后运行 CSI-Time 准确率消融：

```bash
python elder_hampel_training_ablation.py
```

固定 `IQR + min-max + linear64`，比较 no-denoise、Savgol、不同 Hampel 窗口和阈值。只有“替换率/峰值损失明显”与“准确率同步下降”同时出现，才能确认 Hampel 抹除了动作信息。

## 5. ElderAL：Bandpass 高分是否来自短序列绕过

```bash
python elder_bandpass_bypass_diagnostic.py
python elder_bandpass_training_ablation.py
```

第一个脚本逐样本验证源码的 `T < 28` 原样返回条件；第二个固定全部后续步骤，对比 no-denoise 与 Bandpass。

- 大部分样本输出与输入完全相同，且两组准确率接近：说明“Bandpass 表现不错”主要不能归功于实际滤波；
- 只有真正进入滤波的样本很多、Bandpass 又稳定优于 no-denoise，才能说明它确实适合 ElderAL。

## 6. XRF55：采样率是否是唯一原因

```bash
python ablation_xrf55_bandpass_fs.py
```

只改变 `fs=1000/200`，其余固定为 `Bandpass + IQR + z-score + cubic15 + ResNet1D`。如果 200 Hz 稳定更好但仍明显低于 Savgol，说明错误采样率是原因之一，但不是完整根因。

## 7. XRF55：Bandpass 正负输出是否被后续 abs 折叠

先看信号层证据：

```bash
python ablation_xrf55_bandpass_sign.py --diagnostics-only
```

再训练四种链路：

```bash
python ablation_xrf55_bandpass_sign.py
```

四组依次验证：

- 当前 IQR 和归一化都取幅值；
- IQR 保留正负、归一化仍取幅值；
- IQR 和归一化都保留正负；
- 保留正负且不使用 IQR。

若 `signed_iqr_signednorm` 相比当前链路稳定恢复准确率，才能确认“上升/下降方向被 abs 折叠”是核心原因；如果仍不恢复，则还需要继续检查 0.5–50 Hz 频带本身是否过宽。

