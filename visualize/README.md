# BEVDepth 可视化工具

这个目录把 BEVDepth 实验可视化拆成几个独立脚本，主要服务于：

- 预测框 / GT 对比
- 3D box 投影到相机图像
- 按错误类型筛 bad case
- BEV feature、splat 后 BEV、depth 分布、head heatmap 可视化

## 脚本说明

1. `project_boxes_to_cameras.py`
   - 把 nuScenes 3D box 投影到 6 个相机图像。
   - 可选 GT、Pred 或两者同时画。

2. `compare_pred_gt.py`
   - 做预测框/GT 对比。
   - 蓝色是 TP，红色是 FP，黄色是 FN。

3. `mine_bad_cases.py`
   - 按错误类型筛 bad case。
   - 当前支持 FN、FP、远距离、小目标、低可见度等标签。

4. `dump_bev_depth_outputs.py`
   - 加载一个 exp 和 checkpoint，指定一个 `sample_token`，跑一次 forward。
   - 输出模型内部图：输入图像、depth argmax/conf、splat 后 BEV、最终 BEV feature、head heatmap。
   - 可选保存 `internal_outputs.pt`，后续可以继续用 `visualize_feature_depth.py` 画图。

5. `visualize_feature_depth.py`
   - 通用张量可视化小工具，不加载模型、不跑 forward。
   - 用于把已经保存好的 `.npy` 或 `.pt/.pth` 中间张量画成图。
   - 目前 `dump_bev_depth_outputs.py` 复用了其中的 `colorize` 函数；日常分析优先用 `dump_bev_depth_outputs.py`。


## 常用命令

投影 GT + pred：

```bash
python visualize/project_boxes_to_cameras.py \
  --sample-token fd8420396768425eabec9bdddf7e64b6 \
  --pred outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/results_nusc.json \
  --box-source both \
  --score-thr 0.3 \
  --out-dir outputs/vis_boxes
```

预测/GT 对比：

```bash
python visualize/compare_pred_gt.py \
  --sample-token fd8420396768425eabec9bdddf7e64b6 \
  --pred outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/results_nusc.json \
  --score-thr 0.3 \
  --match-dist 2.0 \
  --out-dir outputs/vis_compare
```

挖 bad case：

```bash
python visualize/mine_bad_cases.py \
  --pred outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/results_nusc.json \
  --score-thr 0.3 \
  --match-dist 2.0 \
  --out outputs/bad_cases.json
```

跑一个样本的内部 BEV/depth/head 可视化：

```bash
python visualize/dump_bev_depth_outputs.py \
  --exp bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key \
  --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6/checkpoints/epoch=23-step=10560.ckpt \
  --sample-token 163b70e627854893b88575caf85a56ea \
  --device cuda:0 \
  --out-dir outputs/vis_internal \
  --save-tensors
```

可视化保存的 depth logits：

```bash
python visualize/visualize_feature_depth.py \
  --tensor outputs/debug/depth_pred.pt \
  --kind depth_argmax \
  --camera-index 0 \
  --out outputs/vis_internal/depth_cam0.jpg
```

## 输出怎么看

### 预测框 / GT 对比图

`compare_pred_gt.py` 输出在：

```text
outputs/vis_compare/<sample_token>/<CAMERA>.jpg
```

颜色含义：

- 蓝色 `TP`：预测框和同类别 GT 在 BEV 中心距离内匹配成功。
- 黄色 `FN`：GT 没有匹配到预测框，也就是漏检。
- 红色 `FP`：预测框没有匹配到 GT，也就是误检。

这里的 `FN` 不是“空的地方有框”，空的地方有框是 `FP`。`FN` 是“这里本来有 GT，但是模型没有给出一个同类别、分数够高、中心距离足够近的预测”。

### Bad Case JSON

`mine_bad_cases.py` 输出是一个列表，每一行记录一个具体问题对象，而不是一个 sample 只对应一条记录。

常见字段：

- `sample_token`：这个 bad case 所在的 nuScenes sample。
- `error_type`：`fn`、`fp` 或 `tp_hard_case`。
- `tags`：问题标签，例如 `missed_gt`、`far`、`small`、`low_visibility`。
- `category`：类别。
- `distance`：目标中心到 ego 的距离。
- `volume`：3D box 体积。
- `gt_token` / `pred_score`：用于追踪 GT 或预测结果。

所以同一个 `sample_token` 出现很多次是正常的，表示这个 sample 里有多个漏检、误检或困难 TP。

### 内部特征图

`dump_bev_depth_outputs.py` 输出在：

```text
outputs/vis_internal/<sample_token>/
```

主要文件：

- `input_cam0.jpg` 到 `input_cam5.jpg`：送入模型的 key frame 相机图，已经反归一化；尺寸是模型输入尺寸，例如 `704x256`，不是原始 nuScenes 图像尺寸。
- `depth_argmax_cam0.jpg` 到 `depth_argmax_cam5.jpg`：每个相机在低分辨率特征图上的最大概率 depth bin。
- `depth_conf_cam0.jpg` 到 `depth_conf_cam5.jpg`：每个像素最大 depth 概率，用来看深度预测置信度。
- `splat_sweep0_mean.jpg` / `splat_sweep0_max.jpg`：key frame 经过 `voxel_pooling_train` splat 到 BEV 后的特征响应。
- `splat_sweep1_mean.jpg` / `splat_sweep1_max.jpg`：第二个 sweep splat 到 BEV 后的特征响应；`2key` 配置通常会有两个 sweep。
- `bev_feature_mean.jpg`：多个 sweep 的 BEV feature 拼接后的最终 BEV 特征响应。
- `head_heatmap_task*.jpg`：CenterPoint head 各 task 的 heatmap。
- `head_heatmap_all_tasks.jpg`：所有 task heatmap 取 max 后的总览。
- `internal_outputs.pt`：如果加了 `--save-tensors`，会保存 BEV feature、depth prob、head preds 等张量。

相机编号和 nuScenes 相机的对应关系：

- `cam0`: `CAM_FRONT_LEFT`
- `cam1`: `CAM_FRONT`
- `cam2`: `CAM_FRONT_RIGHT`
- `cam3`: `CAM_BACK_LEFT`
- `cam4`: `CAM_BACK`
- `cam5`: `CAM_BACK_RIGHT`

## 匹配和筛选口径

当前 TP/FP/FN 是为了可视化和 bad case 挖掘写的简化逻辑：

1. 先按 `score_thr` 过滤预测框。
2. 只允许同类别匹配。
3. 用 BEV 中心距离做 greedy matching。
4. 中心距离小于等于 `match_dist` 的预测/GT 记为 TP。
5. 没匹配上的 GT 记为 FN。
6. 没匹配上的预测框记为 FP。

这套逻辑方便定位问题，但不是 nuScenes 官方评估。官方 mAP/NDS 还会使用不同类别距离阈值、多 recall 点、属性/速度/朝向等指标。

调参建议：

- `score_thr` 越高，低置信预测会被过滤，FN 可能变多。
- `score_thr` 越低，候选框变多，FP 可能变多。
- `match_dist` 越小，定位稍偏的框更容易变成 FN/FP。
- 如果图上看起来“明明附近有框却是 FN”，优先检查 `score_thr`、类别是否一致、BEV 中心距离是否超过 `match_dist`。

## 注意

- 预测文件需要是 nuScenes eval 格式的 `results_nusc.json`。
- 现在 TP/FP/FN 使用同类别 BEV 中心距离做简化匹配，默认阈值 `2.0m`；正式报告里可以说明这是用于可视化筛选，不是 nuScenes 官方评估匹配。
- `dump_bev_depth_outputs.py` 会直接跑一次 forward 生成内部图；`visualize_feature_depth.py` 适合对已经保存的 `.pt/.npy` 张量做二次可视化。
- 内部 feature 图只表示响应强弱，不能直接等价为真实障碍物位置；分析 bad case 时最好和相机投影图、预测/GT 对比图一起看。
