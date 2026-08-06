# Robust 组件消融：服务器运行说明

正式入口：`ablation/robust_component_server_ablation.py`

该脚本只在 `ablation/` 下写缓存和结果，不会修改
`SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/src/wsdp` 源码。

## 固定实验口径

两套数据都固定为：

```text
savgol(window=7, polyorder=3)
→ IQR(factor=1.5)
→ phase condition
→ nearest(target_K=15)
→ per-sample z-score amplitude + wrapped phase
→ truncate/pad to 1500
→ mlpmodel
```

- 模型 seed：42
- group split seed：42
- Gait：60 epochs
- Widar：80 epochs
- batch/lr/weight decay：32 / 3e-4 / 1e-3
- 最终输入：`float32 (N,1500,15,6)`，显式 `[amplitude, phase]`
- `robust_first50` 直接调用当前源码 `robust_phase_sanitization()`。

脚本自动兼容两种目录布局：

```text
# 服务器截图中的布局
SDP/
  ablation/
  sdp_dataset/Gait_Dataset/CSI_Gait/
  sdp_dataset/widar_common3/
  SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/

# 本地 outer repo 布局
My_sdp_test/
  ablation/
  sdp_dataset/
  SDP/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main/
```

如自动探测不到，可以显式传 `--data-path`。

## 1. 先检查路径

进入同时包含 `ablation`、`sdp_dataset` 和 WSDP 工程的目录：

```bash
python -u ablation/robust_component_server_ablation.py \
  --dataset gait --suite core --dry-run

python -u ablation/robust_component_server_ablation.py \
  --dataset widar --suite core --dry-run
```

`dry-run` 不读 CSI 内容，也不写结果。确认输出的数据路径、样本数、WSDP
源码路径均正确后再继续。

## 2. 可选的小样本冒烟测试

该命令只验证读取、双进程预处理、源码等价门禁、memmap 和 split，不训练：

```bash
python -u ablation/robust_component_server_ablation.py \
  --dataset gait \
  --conditions robust_first50 \
  --max-samples 18 \
  --preprocess-only \
  --workers 2 \
  --output-root ablation/robust_component_server_smoke
```

任何使用 `--max-samples` 的结果都不能作为正式准确率。

## 3. 正式单 seed 消融

建议不要同时跑 Gait 和 Widar，以免两份前缀缓存同时占用磁盘和 I/O。

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  ablation/robust_component_server_ablation.py \
  --dataset gait --suite core --workers 4

CUDA_VISIBLE_DEVICES=0 python -u \
  ablation/robust_component_server_ablation.py \
  --dataset widar --suite core --workers 4
```

后台运行示例：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u \
  ablation/robust_component_server_ablation.py \
  --dataset gait --suite core --workers 4 \
  > ablation/robust_component_gait.log 2>&1 &
```

`core` 包含八个最有判别力的条件，并按下面的顺序运行：

1. `linear_reference`
2. `robust_first50`
3. `no_calibration`
4. `common_only`
5. `detrend_first50_only`
6. `robust_shared_first50`
7. `robust_window_limited`
8. `robust_fullspan50`

其中 `robust_fullspan50` 在模型最终可见的前 `min(T,1500)` 帧内均匀取 50
个拟合点，用来区分“只用前 50 帧估计不稳”和“逐子载波去趋势本身有害”。

如还要增加“只做独立去趋势、但在模型可见窗口均匀取点”的控制，使用
`--suite full`，它会再运行：

- `detrend_fullspan50_only`

也可以用逗号选择任意子集：

```bash
--conditions linear_reference,robust_first50,common_only
```

## 4. 缓存、空间和续跑

- 全量 Gait 的公共 `savgol+IQR` 前缀预计需要约 40 GiB 磁盘，只生成一次。
- 单个 `(N,1500,15,6)` float32 条件约 9.4 GiB。
- 条件训练成功后，默认删除该条件的 processed memmap；公共前缀仍保留。
- 训练中断时，当前条件的 processed memmap 会保留；重新运行同一命令即可
  从已完成的样本索引继续预处理，不会把整个 memmap 截断重做。
- 训练阶段中断后会复用 processed memmap，但模型会从 epoch 0 重新训练；脚本
  不做 epoch 级续训，以免把“最佳验证 checkpoint”误当成“最后一轮 checkpoint”。
- 只有 `status=ok` 且 checkpoint/history/predictions 的记录大小和内容结构有效
  的条件才会跳过。
- `settings.json` 保存数据、代码、模型和超参数指纹；配置不同会拒绝混入同一
  输出目录。需要改配置时请指定新的 `--output-root`。
- 同一 `output-root + dataset` 有排他锁，防止误启动两份任务并发覆盖 memmap
  和 checkpoint。进程被强杀后若留下锁，先核对锁内 PID 已不存在，再只删除
  对应的 `.gait.lock` 或 `.widar.lock`。
- 正式全量训练时，`--device auto` 若没有看到 CUDA 会直接报错，避免意外在 CPU
  上跑数天；若确实要用 CPU，可显式传 `--device cpu`。
- 启动前会按原始数据约 4 倍估算前缀缓存、当前 processed 数据和 5 GiB 安全
  余量。空间不足会提前终止；只有自行确认空间足够时才使用
  `--skip-disk-check`。
- 如确实需要保留每个条件的 processed 数据，可加 `--keep-processed`，但会快速
  占用数十 GiB。

如果服务器磁盘 I/O 压力大，可把 `--workers 4` 降为 `--workers 2`；不要通过
增加 worker 数盲目抢占公共服务器。

## 5. 结果位置与判据

默认输出：

```text
ablation/robust_component_server_results/
  gait/
  widar/
```

重点文件：

- `summary.csv`：每个条件的验证/测试准确率及相对 Linear/Robust 的 pp 差值。
- `contrasts.json`：预定义的成对因果比较。
- `source_equivalence.json`：自定义 Robust 镜像、nearest 快速路径和完整 tail
  与源码的数值等价门禁；任一不等价都会在训练前终止。
- `split_indices.npz` / `split_metadata.json`：所有条件共享的样本索引和 group
  无泄漏检查。
- `<condition>/seed_42/status.json`：该条件的完整参数和最终准确率。
- `<condition>/seed_42/test_predictions.npz`：同一测试样本上的预测，可继续做
  McNemar 等配对分析。

根因判据：

- `robust_first50 << common_only`：Robust 的逐子载波时间去趋势是主要损失源。
- `robust_shared_first50 ≈ common_only`：破坏来自各子载波使用不同斜率，而非
  普通的共享相位旋转。
- `robust_window_limited > robust_first50`：向 1500 帧长距离外推会放大损失。
- `robust_fullspan50 > robust_first50`：前 50 帧斜率估计不稳是损失来源之一。
- `robust_fullspan50` 仍明显低于 `common_only`：即使在模型可见窗口重新估计，
  沿时间删除逐子载波趋势仍会与动作信息冲突。
