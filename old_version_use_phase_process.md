# 对于xrf55数据集
执行每一步算法后，如果是xrf55数据集并且虚部接近于0
就从复数转为实数数据
如果pipline的归一化中有z-score算法，创建CSIDataset时会设置
    preserve_real_sign = (
        is_amplitude_primary_dataset(DATASET_NAME)
        and isinstance(pipeline_steps, dict)
        and pipeline_steps.get("normalize", {}).get("method") == "z-score"
    )
也就是如果数据集是幅度数据，pipline-step是字典，normalize是z-score这个方法
那么把 preserve_real_sign = True
然后再把这个preserve_real_sign 传给CSIDataset，CSIDataset会进行判断
如果数据已经是实数，就直接保留原数据；
如果数据是虚部接近 0 的复数，就取实部；
如果数据既不是实数，虚部也不接近于0，那么就取幅度abs，但是这个情况在xrf55应该不会出现
总之不会默认取绝对值，从而保留 z-score 归一化后的正负号。



# 对于gait和widar数据集
这两个数据集都是复数CSI，有幅度和相位两个信息。
老版本的测试流程是没有使用相位信息，即使预设中有相位校准calibrate这一类算法
-- linear
-- polynomial
-- stc
-- robust
但是并没有真正到达模型，因为Gait/Widar 的旧版“只取幅度”。
相位校准基本属于“代码执行了，但模型看不到”。老版本源码在test1_gait和test1_widar中
旧版本的代码没有做其他特殊的处理，对于所有预设的算法都是依然会取abs（）
但是widar我们尝试过保留z-score的正负号，效果还是一般。

新的测试文件在test_gait和test_widar中，逻辑是：
有一个use_phase开关，如果使用打开这个开关，则使用相位和幅度两个信息
更深层的逻辑是：
如果使用相位信息，有一个  manual_phase_zscore  这是是否需要单独处理的一个开关
当使用相位信息并且有z-score这个算法的时候就需要单独处理，为什么呢：
```text
    # 判断是否需要单独处理
    # 当使用相位信息并且有z-score这个算法的时候就需要单独处理（理解一下）
    # 原因：源码的z-score会先取幅度然后会计算平均值和标准差，归一化幅度 = (原幅度 - 平均值) / 标准差
    # 如果原数据是复数，源码会把原相位重新乘回去：result = norm_amp * np.exp(1j * phase)
    # 没有直接删除相位信息，但因为 norm_amp 可能是负数，负数会让相位偏移 π
    # 所以如果按照正常的processor处理完成之后
    # 后面在第四步会走 CSIDataset 把划分好的训练验证测试集构造dataloader
    # 使用相位信息并且是复数csi就会走CSIDataset的if use_phase and np.iscomplexobj(data_array):
    # 这样的话执行完z-score会再进行下面这段代码
    # amplitude = np.abs(data_array)
    # phase = np.angle(data_array)
    # data_list = np.concatenate([amplitude, phase], axis=-1)
    # 之前是5*相位，做完z-score之后是-3*相位，这相当于3 * exp(1j * (phase + π))
    # 再进行一次就会abs就会丢失负号信息，和相位信息，变成3幅度，phase = 原相位 + π
    # 负号信息的丢失就会导致和做完z-score之后就是3*相位分不开
    # 因此需要单独处理
```


不会走到 data_list = np.abs(data_array)。
因为前面手动处理已经执行：
np.concatenate([normalized_amplitude, phase], axis=-1)
normalized_amplitude 和 phase 都是实数，所以拼接后的 data_array 也是实数。
进入 CSIDataset 时：
use_phase = False
preserve_real_sign = True
然后走：
elif preserve_real_sign:
    if np.iscomplexobj(data_array):  # False，不进入
        ...
    else:
        data_list = data_array       # 实际走这里
因此最终直接保留：
[z-score带正负号的幅度, 正确相位]
只有手动处理后数据仍然是虚部不为零的复数时，才会执行 np.abs(data_array)；正常的 manual_phase_zscore 流程不会出现这种情况。
