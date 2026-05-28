# BEVDepth 简历实验清单

## README/代码中已确认的信息

- 官方训练命令：`python [EXP_PATH] --amp_backend native -b 8 --gpus 8`。
- 官方评估命令：`python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8`。
- 默认训练轮数是 24 epochs，来源于 `bevdepth/exps/base_cli.py`：
  `max_epochs=extra_trainer_config_args.get('epochs', 24)`。
- `20e_cbgs_2key_da` 系列实验会显式覆盖为 20 epochs。
- README benchmark 主要记录 `mAP` 和 `NDS`；主 baseline 是
  `bev_depth_lss_r50_256x704_128x128_24e_2key.py`，README 指标为
  mAP `0.3304`、NDS `0.4355`。
- EMA 实验需要单独注意：README 提到 EMA ckpt 保存参数与训练阶段使用参数不同，
  因此 EMA 实验不支持从 ckpt 恢复训练。

## 目标

为简历描述形成可验证证据：

> 在 nuScenes 上复现并调试多相机 BEV 3D 检测流程，分析数据加载、坐标系转换、
> 显存占用和环境兼容问题；设计图像分辨率、depth bins、学习率等对比实验，
> 记录 mAP、NDS、loss 曲线和收敛稳定性；开发可视化与 bad case 分析流程。

## 阶段 0：复现环境与记录规范

- [ ] 记录机器和环境：
  - GPU 型号、driver、CUDA toolkit、`torch`、`torchvision`、`mmcv-full`、
    `mmdet`、`mmdet3d`、`pytorch-lightning`、`numpy`。
  - 当前已知情况：RTX 4090 + cu113 可以禁用 cuDNN 做 debug 跑通，但不是理想训练环境。
  - 改成cu117 可以用cuDNN，但是速度无明显提高，说明不是瓶颈。
- [ ] 记录数据路径和生成的 info 文件：
  - `data/nuScenes/nuscenes_infos_train.pkl`
  - `data/nuScenes/nuscenes_infos_val.pkl`
  - `data/nuScenes/nuscenes_infos_test.pkl`
- [ ] 每个实验保存完整运行命令、git commit hash、改动文件、配置文件路径。
- [ ] 每个实验统一放在 `outputs/<experiment_name>/` 下，并归档：
  - `hparams.yaml`
  - TensorBoard event 文件
  - checkpoint 路径
  - eval 指标
  - 失败原因或兼容性修复记录

## 阶段 1：代码理解与冒烟测试

- [ ] 数据加载链路：
  - 在 `NuscDetDataset.__getitem__` 打断点。
  - 记录 `sweep_imgs`、`mats`、`gt_boxes`、`gt_labels`、`depth_labels` 的 shape。
  - 解释 lidar-to-image depth label 生成过程：
    原始 lidar `(N, 5)` -> 保留 `(x, y, z, intensity)` -> 投影到相机
    -> 稀疏 `[u, v, depth]` -> depth map。
- [ ] 坐标系转换链路：
  - 在 `get_image`、`get_lidar_depth`、`map_pointcloud_to_image`、
    `BaseLSSFPN.get_geometry` 打断点。
  - 记录 `sensor2ego`、`ego2global`、`intrinsics`、`ida`、`bda`、
    `sensor2sensor` 各自作用。
- [ ] 模型模块链路：
  - `BaseBEVDepth.forward`
  - `BaseLSSFPN.get_cam_feats`
  - `BaseLSSFPN._forward_depth_net`
  - `BaseLSSFPN.get_geometry`
  - `voxel_pooling_train`
  - `BEVDepthHead.forward/get_targets/loss`
- [ ] 跑单元测试/冒烟测试：
  - `python -m pytest test/test_ops/test_voxel_pooling.py -q`
  - `python -m pytest test/test_dataset/test_nusc_mv_det_dataset.py -q`
- [ ] 跑 1 个 batch 训练 debug：
  - `-b 1 --gpus 1 --precision 32 --max_epochs 1 --limit_train_batches 1 --limit_val_batches 0`
  - 记录是否需要禁用 cuDNN，以及原因。

## 阶段 2：Baseline 复现

- [ ] baseline 训练/调试目标：
  - `bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py`
  - 官方设置：24 epochs，8 GPUs，每卡 batch size 8。
  - 本地可以用更小配置跑通流程，但必须记录与官方设置的差异。
- [ ] baseline 评估：
  - 使用 `-e --ckpt_path` 评估 checkpoint。
  - 记录 mAP、NDS、mATE、mASE、mAOE、mAVE、mAAE。
- [ ] 与 README baseline 对齐：
  - README mAP `0.3304`，NDS `0.4355`。
  - 记录差异来源：硬件、batch size、训练轮数、AMP、依赖版本、是否完整训练等。
- [ ] loss 曲线记录：
  - 跟踪 `detection_loss`、`depth_loss`、total loss、learning rate。
  - 导出 TensorBoard 截图或 CSV。

## 阶段 3：对比实验

### 图像分辨率

- [ ] 低分辨率：
  - `bev_depth_lss_r50_256x704_128x128_24e_2key.py`
  - 记录显存、速度、mAP、NDS。
- [ ] 中分辨率：
  - `bev_depth_lss_r50_512x1408_128x128_24e_2key.py`
  - 记录显存、速度、mAP、NDS。
- [ ] 高分辨率：
  - `bev_depth_lss_r50_640x1600_128x128_24e_2key.py`
  - 运行前仔细检查 exp name 和输出路径。
  - 记录显存、速度、mAP、NDS。
- [ ] 总结：
  - 更高图像分辨率是否改善小目标/远距离目标？
  - 显存和收敛成本增加多少？

### Depth Bins

- [ ] baseline depth bins：
  - `d_bound=[2.0, 58.0, 0.5]`，depth channels = 112。
- [ ] 更粗 depth bins：
  - 尝试 bin size `1.0`，例如 `[2.0, 58.0, 1.0]`。
  - 记录显存、速度、depth loss、mAP、NDS。
- [ ] 更细或更远 depth bins：
  - 只有显存允许时再尝试更小 bin 或更大的 max depth。
  - 记录对远距离目标和 depth loss 稳定性的影响。
- [ ] 总结：
  - 比较 depth 离散精度、计算成本和检测效果之间的权衡。

### 学习率

- [ ] baseline LR：
  - 代码计算方式：`lr = 2e-4 / 64 * batch_size_per_device * gpus`。
- [ ] 低学习率：
  - 尝试 `0.5x` effective LR。
  - 记录收敛稳定性和最终指标。
- [ ] 高学习率：
  - 尝试 `2x` effective LR。
  - 观察 loss spike、NaN、depth loss 不稳定等问题。
- [ ] 总结：
  - 绘制 loss 曲线，对比收敛速度和稳定性。

### EMA / CBGS / DA

- [ ] EMA 对比：
  - 比较 non-EMA 和 EMA 的 checkpoint/eval 行为。
  - 记录 README 中 EMA 不能从 ckpt resume 的限制。
- [ ] CBGS + DA 对比：
  - `bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.py`
  - 官方训练长度：20 epochs。
  - README mAP `0.3484`，NDS `0.4805`。
- [ ] EMA + CBGS + DA：
  - README mAP `0.3589`，NDS `0.4797`。
  - 注意 README 中 EMA 提升 mAP，但 NDS 不一定同步提升。

## 阶段 4：可视化与 Bad Case 分析

- [ ] 3D box 投影：
  - 将 GT 和预测 3D box 投影到相机图像。
  - 保存代表性场景的 side-by-side 可视化。
- [ ] BEV 可视化：
  - 在 BEV 平面绘制 GT box 和预测 box。
  - 按类别和 score 区分颜色。
- [ ] Depth 可视化：
  - 保存稀疏 lidar depth map。
  - 保存预测 depth 分布或 argmax depth。
  - 与原图和 lidar 投影深度对比。
- [ ] BEV feature 可视化：
  - hook voxel pooling 后或 BEV neck 后的 feature。
  - 保存 channel mean/max heatmap。
- [ ] Bad case 标签：
  - 远距离
  - 小目标
  - 遮挡
  - 图像边界截断
  - 夜晚/雨天/低照度
  - 拥挤场景
  - 类别混淆
- [ ] 每个 bad case 记录：
  - sample token
  - scene name
  - camera view
  - 指标症状
  - 可能原因
  - 可能改进方向

## 阶段 5：结果表模板

| ID | 实验 | Epochs | 图像尺寸 | Depth Bins | LR Scale | EMA | CBGS/DA | Batch/GPU | mAP | NDS | 峰值显存 | 备注 |
| --- | --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| E00 | smoke_1batch | 1 step | 256x704 | 2-58/0.5 | 1.0 | 否 | 否 | 1 | - | - | - | debug only |
| E01 | baseline_256x704 | 24 | 256x704 | 2-58/0.5 | 1.0 | 否 | 否 | 8 | 0.3304 ref | 0.4355 ref | - | README reference |
| E02 | baseline_256x704_ema | 24 | 256x704 | 2-58/0.5 | 1.0 | 是 | 否 | 8 | 0.3329 ref | 0.4409 ref | - | README reference |
| E03 | cbgs_da | 20 | 256x704 | 2-58/0.5 | 1.0 | 否 | 是 | 8 | 0.3484 ref | 0.4805 ref | - | README reference |
| E04 | res_512x1408 | 24 | 512x1408 | 2-58/0.5 | 1.0 | 否 | 否 | TBD | TBD | TBD | TBD | 分辨率对比 |
| E05 | res_640x1600 | 24 | 640x1600 | 2-58/0.5 | 1.0 | 否 | 否 | TBD | TBD | TBD | TBD | 分辨率对比 |
| E06 | depth_bin_1m | TBD | 256x704 | 2-58/1.0 | 1.0 | 否 | 否 | TBD | TBD | TBD | TBD | depth bin 对比 |
| E07 | lr_0.5x | TBD | 256x704 | 2-58/0.5 | 0.5 | 否 | 否 | TBD | TBD | TBD | TBD | 学习率对比 |
| E08 | lr_2x | TBD | 256x704 | 2-58/0.5 | 2.0 | 否 | 否 | TBD | TBD | TBD | TBD | 学习率对比 |

## 简历证据产物

- [ ] 环境与兼容性记录：
  - 说明 PyTorch/CUDA/OpenMMLab 版本兼容、CUDA extension 重编译、
    numpy/networkx 修复、RTX 4090 cuDNN workaround。
- [ ] 复现实验日志：
  - 命令、配置、指标、输出路径、checkpoint。
- [ ] 对比实验报告：
  - 图像分辨率/depth bins/学习率表格。
  - loss 曲线图。
  - mAP/NDS 对比表。
- [ ] 可视化报告：
  - 相机视角 3D box 投影。
  - BEV 预测框 vs GT。
  - depth map 可视化。
  - BEV feature heatmap。
- [ ] Bad case 报告：
  - 至少 20 个代表性案例，按失败类型分组。
  - 包含具体 sample token 和截图。



