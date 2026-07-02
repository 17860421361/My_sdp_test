# Recommended Algorithm Combinations

本文档列出建议优先测试的 4 个数据集算法组合。目标不是穷举所有数学组合，而是保留更可能稳定进入现有深度模型训练流程的组合：

```text
denoise -> calibrate -> normalize -> optional interpolate
```

暂不把 `extract_features`、`detect` 纳入全量训练组合，因为它们可能改变数据形态，导致很多模型不能直接接收输出。`outliers` 和 `agc` 也先不放入第一轮搜索，避免组合空间过大。

## Method 参数约定

表格中的 method 名称对应如下 `pipeline_steps` 配置。`none` 表示该步骤不写入 `pipeline_steps`。

| Step | Method | pipeline_steps 写法 |
|---|---|---|
| denoise | wavelet | `{"denoise": {"method": "wavelet"}}` |
| denoise | butterworth | `{"denoise": {"method": "butterworth", "order": 4, "cutoff": 0.25}}` |
| denoise | savgol | `{"denoise": {"method": "savgol", "window_length": 11, "polyorder": 3}}` |
| calibrate | linear | `{"calibrate": {"method": "linear"}}` |
| calibrate | polynomial | `{"calibrate": {"method": "polynomial", "degree": 2}}` |
| calibrate | stc | `{"calibrate": {"method": "stc"}}` |
| calibrate | robust | `{"calibrate": {"method": "robust"}}` |
| normalize | z-score | `{"normalize": {"method": "z-score"}}` |
| normalize | min-max | `{"normalize": {"method": "min-max"}}` |
| interpolate | none | 不写入 `interpolate` |
| interpolate | cubic30 | `{"interpolate": {"method": "cubic", "target_K": 30}}` |
| interpolate | cubic64 | `{"interpolate": {"method": "cubic", "target_K": 64}}` |

## 数量汇总

| Dataset | 组合数 |
|---|---:|
| xrf55 | 18 |
| widar | 54 |
| gait | 36 |
| elderAL | 18 |
| Total | 126 |

## xrf55: 18 组

XRF55 是 amplitude-primary 数据集，优先测试 `denoise + normalize`，同时加入少量 `calibrate` 做对照。这里不加入 `interpolate`。

| ID | denoise | calibrate | normalize | interpolate |
|---|---|---|---|---|
| xrf55-01 | wavelet | none | z-score | none |
| xrf55-02 | wavelet | none | min-max | none |
| xrf55-03 | wavelet | linear | z-score | none |
| xrf55-04 | wavelet | linear | min-max | none |
| xrf55-05 | wavelet | robust | z-score | none |
| xrf55-06 | wavelet | robust | min-max | none |
| xrf55-07 | butterworth | none | z-score | none |
| xrf55-08 | butterworth | none | min-max | none |
| xrf55-09 | butterworth | linear | z-score | none |
| xrf55-10 | butterworth | linear | min-max | none |
| xrf55-11 | butterworth | robust | z-score | none |
| xrf55-12 | butterworth | robust | min-max | none |
| xrf55-13 | savgol | none | z-score | none |
| xrf55-14 | savgol | none | min-max | none |
| xrf55-15 | savgol | linear | z-score | none |
| xrf55-16 | savgol | linear | min-max | none |
| xrf55-17 | savgol | robust | z-score | none |
| xrf55-18 | savgol | robust | min-max | none |

## widar: 54 组

Widar 是 gesture 分类，位置、方向和接收器会影响分布，建议重点覆盖 `denoise + calibrate + normalize + interpolate`。

| ID | denoise | calibrate | normalize | interpolate |
|---|---|---|---|---|
| widar-01 | wavelet | linear | z-score | none |
| widar-02 | wavelet | linear | z-score | cubic30 |
| widar-03 | wavelet | linear | z-score | cubic64 |
| widar-04 | wavelet | linear | min-max | none |
| widar-05 | wavelet | linear | min-max | cubic30 |
| widar-06 | wavelet | linear | min-max | cubic64 |
| widar-07 | wavelet | stc | z-score | none |
| widar-08 | wavelet | stc | z-score | cubic30 |
| widar-09 | wavelet | stc | z-score | cubic64 |
| widar-10 | wavelet | stc | min-max | none |
| widar-11 | wavelet | stc | min-max | cubic30 |
| widar-12 | wavelet | stc | min-max | cubic64 |
| widar-13 | wavelet | robust | z-score | none |
| widar-14 | wavelet | robust | z-score | cubic30 |
| widar-15 | wavelet | robust | z-score | cubic64 |
| widar-16 | wavelet | robust | min-max | none |
| widar-17 | wavelet | robust | min-max | cubic30 |
| widar-18 | wavelet | robust | min-max | cubic64 |
| widar-19 | butterworth | linear | z-score | none |
| widar-20 | butterworth | linear | z-score | cubic30 |
| widar-21 | butterworth | linear | z-score | cubic64 |
| widar-22 | butterworth | linear | min-max | none |
| widar-23 | butterworth | linear | min-max | cubic30 |
| widar-24 | butterworth | linear | min-max | cubic64 |
| widar-25 | butterworth | stc | z-score | none |
| widar-26 | butterworth | stc | z-score | cubic30 |
| widar-27 | butterworth | stc | z-score | cubic64 |
| widar-28 | butterworth | stc | min-max | none |
| widar-29 | butterworth | stc | min-max | cubic30 |
| widar-30 | butterworth | stc | min-max | cubic64 |
| widar-31 | butterworth | robust | z-score | none |
| widar-32 | butterworth | robust | z-score | cubic30 |
| widar-33 | butterworth | robust | z-score | cubic64 |
| widar-34 | butterworth | robust | min-max | none |
| widar-35 | butterworth | robust | min-max | cubic30 |
| widar-36 | butterworth | robust | min-max | cubic64 |
| widar-37 | savgol | linear | z-score | none |
| widar-38 | savgol | linear | z-score | cubic30 |
| widar-39 | savgol | linear | z-score | cubic64 |
| widar-40 | savgol | linear | min-max | none |
| widar-41 | savgol | linear | min-max | cubic30 |
| widar-42 | savgol | linear | min-max | cubic64 |
| widar-43 | savgol | stc | z-score | none |
| widar-44 | savgol | stc | z-score | cubic30 |
| widar-45 | savgol | stc | z-score | cubic64 |
| widar-46 | savgol | stc | min-max | none |
| widar-47 | savgol | stc | min-max | cubic30 |
| widar-48 | savgol | stc | min-max | cubic64 |
| widar-49 | savgol | robust | z-score | none |
| widar-50 | savgol | robust | z-score | cubic30 |
| widar-51 | savgol | robust | z-score | cubic64 |
| widar-52 | savgol | robust | min-max | none |
| widar-53 | savgol | robust | min-max | cubic30 |
| widar-54 | savgol | robust | min-max | cubic64 |

## gait: 36 组

Gait 是 user_id 分类，建议主要比较不同相位校准和归一化策略；插值只保留 `none` 与 `cubic30` 两档。

| ID | denoise | calibrate | normalize | interpolate |
|---|---|---|---|---|
| gait-01 | wavelet | linear | z-score | none |
| gait-02 | wavelet | linear | z-score | cubic30 |
| gait-03 | wavelet | linear | min-max | none |
| gait-04 | wavelet | linear | min-max | cubic30 |
| gait-05 | wavelet | polynomial | z-score | none |
| gait-06 | wavelet | polynomial | z-score | cubic30 |
| gait-07 | wavelet | polynomial | min-max | none |
| gait-08 | wavelet | polynomial | min-max | cubic30 |
| gait-09 | wavelet | stc | z-score | none |
| gait-10 | wavelet | stc | z-score | cubic30 |
| gait-11 | wavelet | stc | min-max | none |
| gait-12 | wavelet | stc | min-max | cubic30 |
| gait-13 | butterworth | linear | z-score | none |
| gait-14 | butterworth | linear | z-score | cubic30 |
| gait-15 | butterworth | linear | min-max | none |
| gait-16 | butterworth | linear | min-max | cubic30 |
| gait-17 | butterworth | polynomial | z-score | none |
| gait-18 | butterworth | polynomial | z-score | cubic30 |
| gait-19 | butterworth | polynomial | min-max | none |
| gait-20 | butterworth | polynomial | min-max | cubic30 |
| gait-21 | butterworth | stc | z-score | none |
| gait-22 | butterworth | stc | z-score | cubic30 |
| gait-23 | butterworth | stc | min-max | none |
| gait-24 | butterworth | stc | min-max | cubic30 |
| gait-25 | savgol | linear | z-score | none |
| gait-26 | savgol | linear | z-score | cubic30 |
| gait-27 | savgol | linear | min-max | none |
| gait-28 | savgol | linear | min-max | cubic30 |
| gait-29 | savgol | polynomial | z-score | none |
| gait-30 | savgol | polynomial | z-score | cubic30 |
| gait-31 | savgol | polynomial | min-max | none |
| gait-32 | savgol | polynomial | min-max | cubic30 |
| gait-33 | savgol | stc | z-score | none |
| gait-34 | savgol | stc | z-score | cubic30 |
| gait-35 | savgol | stc | min-max | none |
| gait-36 | savgol | stc | min-max | cubic30 |

## elderAL: 18 组

ElderAL 序列较短，先不加入插值，重点比较去噪、相位校准与归一化。

| ID | denoise | calibrate | normalize | interpolate |
|---|---|---|---|---|
| elderAL-01 | wavelet | linear | z-score | none |
| elderAL-02 | wavelet | linear | min-max | none |
| elderAL-03 | wavelet | stc | z-score | none |
| elderAL-04 | wavelet | stc | min-max | none |
| elderAL-05 | wavelet | robust | z-score | none |
| elderAL-06 | wavelet | robust | min-max | none |
| elderAL-07 | butterworth | linear | z-score | none |
| elderAL-08 | butterworth | linear | min-max | none |
| elderAL-09 | butterworth | stc | z-score | none |
| elderAL-10 | butterworth | stc | min-max | none |
| elderAL-11 | butterworth | robust | z-score | none |
| elderAL-12 | butterworth | robust | min-max | none |
| elderAL-13 | savgol | linear | z-score | none |
| elderAL-14 | savgol | linear | min-max | none |
| elderAL-15 | savgol | stc | z-score | none |
| elderAL-16 | savgol | stc | min-max | none |
| elderAL-17 | savgol | robust | z-score | none |
| elderAL-18 | savgol | robust | min-max | none |

## Python 生成提示

把表格中的一行转换成 `pipeline_steps` 时，按以下规则拼接即可：

```python
pipeline_steps = {}

if denoise != "none":
    pipeline_steps["denoise"] = DENOISE_CONFIGS[denoise]

if calibrate != "none":
    pipeline_steps["calibrate"] = CALIBRATE_CONFIGS[calibrate]

if normalize != "none":
    pipeline_steps["normalize"] = NORMALIZE_CONFIGS[normalize]

if interpolate != "none":
    pipeline_steps["interpolate"] = INTERPOLATE_CONFIGS[interpolate]
```
