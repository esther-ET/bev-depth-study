# 1 基准实验
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py -b 4 --gpus 1 --accumulate_grad_batches 16
1. 原图大小的eval
```
Evaluating bboxes of img_bbox
mAP: 0.3141
mATE: 0.7322
mASE: 0.2795
mAOE: 0.6349
mAVE: 0.5500
mAAE: 0.2159
NDS: 0.4158
Eval time: 38.3s（官方计算metric的耗时）

Per-class results:

Object Class AP ATE ASE AOE AVE AAE

car 0.485 0.553 0.168 0.209 0.577 0.228

truck 0.254 0.730 0.224 0.254 0.516 0.217

bus 0.362 0.690 0.199 0.186 1.049 0.257

trailer 0.143 1.162 0.254 0.783 0.520 0.123

construction_vehicle 0.069 1.125 0.505 1.291 0.127 0.406

pedestrian 0.276 0.762 0.295 0.977 0.581 0.295

motorcycle 0.315 0.648 0.264 0.859 0.783 0.189

bicycle 0.311 0.579 0.260 0.922 0.247 0.011

traffic_cone 0.440 0.533 0.350 nan nan nan

barrier 0.486 0.538 0.276 0.233 nan nan

  

### 测试耗时

- 总测试时间：457.26s，约 7.6min 包含 dataloader + batch_to_device + model test_step + 后处理 + test_epoch_end

- 测试 batch 数：1505

🫪- 单 batch test_step 平均耗时：0.155s 🫪可以作为工程上的端到端 batch 推理耗时，每个 batch 的推理 + 单 batch 后处理耗时，0.15536s / 4 ≈ 0.03884s / sample，即约 25.7 FPS ，不过是 按“一个 sample = 6 相机一帧”算的。

- GPU 数据搬运 batch_to_device 平均耗时：0.063s，占总时间约 20.6%

- test_epoch_end / nuScenes 评估耗时：115.28s，占总时间约 25.2% 包含收集预测结果、格式化 bbox、写 json、调用 nuScenes evaluator、汇总结果等

- 主要耗时来自模型推理后处理和最终 nuScenes evaluation，dataloader 初始化耗时较小。
```

2. 用这个点测试小图像
AP: 0.2520 mATE: 0.8423 mASE: 0.2806 mAOE: 0.6508 mAVE: 0.5512 mAAE: 0.2235 NDS: 0.3712   
Eval time: 40s
Per-class results:                                                                                                
Object Class            AP      ATE     ASE     AOE     AVE     AAE                                               
car                     0.389   0.720   0.175   0.231   0.585   0.229                                             
truck                   0.191   0.880   0.231   0.262   0.551   0.232                                             
bus                     0.287   0.828   0.212   0.217   1.115   0.284                                             
trailer                 0.121   1.138   0.231   0.665   0.401   0.142                                             
construction_vehicle    0.048   1.079   0.506   1.491   0.130   0.396                                             
pedestrian              0.221   0.825   0.296   0.974   0.589   0.300                                             
motorcycle              0.244   0.827   0.263   0.873   0.782   0.191                                             
bicycle                 0.202   0.861   0.260   0.920   0.255   0.015                                             
traffic_cone            0.399   0.580   0.352   nan     nan     nan                                               
barrier                 0.418   0.685   0.281   0.225   nan     nan

b=4 显存很小4000多
- test_step = 0.1115 s/batch ≈ 27.9 ms/sample
- total test wall time = 377.0 s
- test_epoch_end = 116.7 s

3. 用大一点的图像测试
AP: 0.2754  mATE: 0.8241  mASE: 0.2834 mAOE: 0.6534  mAVE: 0.5479  mAAE: 0.2275 NDS: 0.3841   Eval time: 39.2s 
Per-class results:                                                                             
Object Class            AP      ATE     ASE     AOE     AVE     AAE                            
car                     0.439   0.661   0.169   0.214   0.572   0.232                          
truck                   0.218   0.817   0.230   0.256   0.508   0.226                          
bus                     0.275   0.824   0.202   0.202   0.995   0.271                          
trailer                 0.099   1.204   0.266   0.743   0.514   0.103                          
construction_vehicle    0.053   1.105   0.512   1.367   0.127   0.487                          
pedestrian              0.262   0.852   0.300   1.009   0.590   0.300                          
motorcycle              0.265   0.793   0.264   0.825   0.851   0.192                          
bicycle                 0.292   0.699   0.264   1.020   0.226   0.008                          
traffic_cone            0.400   0.646   0.345   nan     nan     nan                            
barrier                 0.450   0.641   0.282   0.244   nan     nan

test_step mean = 0.22679 s / batch
batch size = 4
0.22679 / 4 = 0.0567 s

frustum 点数量按 112 * H_feat * W_feat * 6 cameras 走，所以：
192x640: 112 * 12 * 40 * 6 = 322,560
256x704: 112 * 16 * 44 * 6 = 473,088
320x864: 112 * 20 * 54 * 6 = 725,760

总结下：
test size | test_step s/batch | ms/sample | total time
192x640   | 0.1115            | 27.9      | 377.0s.     mAP: 0.2520 NDS:0.3712
256x704   | 0.1554            | 38.8      | 457.3s      mAP: 0.3141 NDS: 0.4158
320x864   | 0.2268            | 56.7      | 625.0s      mAP: 0.2754 NDS: 0.3841

# 2 小图 depth
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_192x640_128x128_24e_2key.py -b 4 --gpus 1 --accumulate_grad_batches 16




ckpt:
`/home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_192x640_128x128_24e_2key/lightning_logs/version_1/checkpoints/epoch=23-step=10560.ckpt`

说明：这组测试是在两个学习率训练进程同时占用 GPU/CPU 的情况下跑的，所以 profiler 里的 total time、test_step time 明显偏慢；mAP/NDS 指标正常可用于实验对比，耗时只作为本次机器负载下的参考。

1. 用小图训练点测试 192x640

```
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_192x640_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_192x640_128x128_24e_2key/lightning_logs/version_1/checkpoints/epoch=23-step=10560.ckpt
```

AP: 0.2671  mATE: 0.7483  mASE: 0.2908  mAOE: 0.7466  mAVE: 0.6400  mAAE: 0.2325  NDS: 0.3677
Eval time: 288.5s

Per-class results:

Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.444   0.565   0.174   0.262   0.622   0.229
truck                   0.186   0.782   0.249   0.312   0.566   0.245
bus                     0.280   0.712   0.228   0.237   1.213   0.291
trailer                 0.090   1.110   0.243   1.033   0.862   0.140
construction_vehicle    0.033   1.080   0.554   1.445   0.124   0.431
pedestrian              0.226   0.793   0.298   0.991   0.594   0.311
motorcycle              0.287   0.672   0.257   1.016   0.876   0.202
bicycle                 0.280   0.602   0.274   1.195   0.261   0.010
traffic_cone            0.399   0.591   0.351   nan     nan     nan
barrier                 0.445   0.575   0.279   0.227   nan     nan

- total time = 2005.0s
- test_step = 0.25243s/batch ≈ 63.1ms/sample
- batch_to_device = 0.03002s/batch
- test_epoch_end = 766.87s

2. 用小图训练点测试 256x704

```
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_192x640_128x128_24e_2key/lightning_logs/version_1/checkpoints/epoch=23-step=10560.ckpt
```

AP: 0.2194  mATE: 0.8665  mASE: 0.3005  mAOE: 0.7811  mAVE: 0.6153  mAAE: 0.2410  NDS: 0.3293
Eval time: 314.6s

Per-class results:

Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.379   0.712   0.178   0.276   0.625   0.234
truck                   0.138   0.931   0.269   0.352   0.617   0.248
bus                     0.206   0.779   0.253   0.258   1.197   0.303
trailer                 0.060   1.138   0.285   1.250   0.547   0.064
construction_vehicle    0.027   1.056   0.535   1.374   0.136   0.546
pedestrian              0.200   0.940   0.302   1.038   0.609   0.332
motorcycle              0.228   0.859   0.261   1.067   0.918   0.187
bicycle                 0.206   0.819   0.284   1.175   0.274   0.014
traffic_cone            0.353   0.709   0.357   nan     nan     nan
barrier                 0.396   0.724   0.280   0.240   nan     nan

- total time = 2136.8s
- test_step = 0.31234s/batch ≈ 78.1ms/sample
- batch_to_device = 0.04722s/batch
- test_epoch_end = 818.39s

3. 用小图训练点测试 320x864

```
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_320x864_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_192x640_128x128_24e_2key/lightning_logs/version_1/checkpoints/epoch=23-step=10560.ckpt
```

AP: 0.2096  mATE: 0.8516  mASE: 0.3125  mAOE: 0.7913  mAVE: 0.6709  mAAE: 0.2523  NDS: 0.3170
Eval time: 294.7s

Per-class results:

Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.384   0.664   0.181   0.288   0.708   0.247
truck                   0.133   0.882   0.295   0.425   0.663   0.231
bus                     0.194   0.767   0.265   0.257   1.396   0.330
trailer                 0.032   1.126   0.341   1.252   0.597   0.058
construction_vehicle    0.026   1.059   0.534   1.421   0.129   0.580
pedestrian              0.182   0.941   0.309   1.061   0.617   0.347
motorcycle              0.204   0.848   0.264   0.999   0.977   0.219
bicycle                 0.208   0.844   0.288   1.171   0.279   0.006
traffic_cone            0.353   0.671   0.360   nan     nan     nan
barrier                 0.382   0.714   0.289   0.248   nan     nan

- total time = 2233.7s
- test_step = 0.41338s/batch ≈ 103.3ms/sample
- batch_to_device = 0.07017s/batch
- test_epoch_end = 787.75s

总结：

test size | test_step s/batch | ms/sample | total time | mAP | NDS
192x640   | 0.2524            | 63.1      | 2005.0s    | 0.2671 | 0.3677
256x704   | 0.3123            | 78.1      | 2136.8s    | 0.2194 | 0.3293
320x864   | 0.4134            | 103.3     | 2233.7s    | 0.2096 | 0.3170

观察：
- 小图训练点在小图测试上最好，换到 256x704 和 320x864 都掉点。
- 这说明模型对训练/测试图像尺度一致性比较敏感；小图训练后，并不会因为测试图像更大就自然变好。
- 和 #1 的 256x704 训练点对比，256 训练点在 256x704 上最好；小图训练点在 192x640 上最好，这个现象比较符合“尺度分布/深度几何网格过拟合当前训练设置”的解释。
- 本组耗时受并行训练进程影响较大，不建议和 #1 的耗时做严格横向对比；更适合比较同一轮测试内 192/256/320 的相对变化。


# 3 小学习率 depth2
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -b 4 --gpus 1 --accumulate_grad_batches 16 --lr_scale 0.5

ckpt:
`/home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_9/checkpoints/epoch=23-step=10560.ckpt`

测试命令：
```
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_9/checkpoints/epoch=23-step=10560.ckpt
```

AP: 0.3078  mATE: 0.7187  mASE: 0.2892  mAOE: 0.7522  mAVE: 0.5999  mAAE: 0.2247  NDS: 0.3955
Eval time: 63.6s

Per-class results:

Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.472   0.567   0.174   0.279   0.650   0.241
truck                   0.247   0.752   0.239   0.311   0.619   0.238
bus                     0.368   0.689   0.224   0.286   1.135   0.261
trailer                 0.143   1.064   0.249   1.140   0.538   0.176
construction_vehicle    0.062   1.048   0.513   1.459   0.128   0.370
pedestrian              0.265   0.762   0.299   0.999   0.616   0.298
motorcycle              0.314   0.665   0.268   0.924   0.853   0.202
bicycle                 0.295   0.551   0.272   1.117   0.261   0.012
traffic_cone            0.425   0.534   0.364   nan     nan     nan
barrier                 0.486   0.554   0.290   0.253   nan     nan

- total time = 720.34s
- test_step = 0.16664s/batch ≈ 41.7ms/sample
- batch_to_device = 0.01892s/batch
- test_epoch_end = 183.4s

# 4 大学习率 depth3
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -b 4 --gpus 1 --accumulate_grad_batches 16 --lr_scale 2.0

ckpt:
`/home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_11/checkpoints/epoch=23-step=10560.ckpt`

测试命令：
```
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_11/checkpoints/epoch=23-step=10560.ckpt
```

AP: 0.3215  mATE: 0.7050  mASE: 0.2780  mAOE: 0.6164  mAVE: 0.5071  mAAE: 0.2265  NDS: 0.4275
Eval time: 62.9s

Per-class results:

Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.494   0.544   0.166   0.184   0.528   0.227
truck                   0.263   0.755   0.223   0.218   0.465   0.209
bus                     0.379   0.656   0.205   0.155   0.926   0.250
trailer                 0.166   1.035   0.245   0.551   0.553   0.178
construction_vehicle    0.060   1.047   0.522   1.382   0.129   0.432
pedestrian              0.278   0.754   0.296   0.933   0.557   0.293
motorcycle              0.315   0.672   0.259   0.875   0.661   0.217
bicycle                 0.327   0.539   0.253   1.034   0.238   0.006
traffic_cone            0.441   0.538   0.339   nan     nan     nan
barrier                 0.490   0.509   0.271   0.216   nan     nan

- total time = 726.2s
- test_step = 0.16251s/batch ≈ 40.6ms/sample
- batch_to_device = 0.01733s/batch
- test_epoch_end = 225.88s

学习率实验总结：

| exp | lr_scale | mAP | NDS | 主要观察 |
|---|---:|---:|---:|---|
| 基准 | 1.0 | 0.3141 | 0.4158 | 原始设置 |
| 小学习率 depth2 | 0.5 | 0.3078 | 0.3955 | mAP/NDS 都低于基准，尤其 AOE、AVE 变差 |
| 大学习率 depth3 | 2.0 | 0.3215 | 0.4275 | 三组里最好，mAP/NDS 都高于基准，ATE/AOE/AVE 也更好 |

结论：
- 当前这组实验里，`lr_scale=2.0` 效果最好，`lr_scale=0.5` 效果最差。
- 说明这个配置在 24 epoch 内可能还没有被原始学习率充分优化，稍微放大学习率有利于收敛。
- 但这不是说学习率越大越好；还需要看训练曲线是否抖动、是否出现 loss spike、最终 epoch 前是否已经过拟合。

除了 mAP/NDS，还可以分析：
- loss 曲线：总 loss、heatmap loss、bbox/reg loss、depth loss 是否下降更快，后期是否震荡。
- validation 指标随 epoch 的变化：哪个学习率更早达到较好 NDS，哪个后期更稳。
- 各类指标拆分：mATE 看位置，mASE 看尺寸，mAOE 看朝向，mAVE 看速度；大学习率这轮主要在 AOE/AVE 上更好。
- per-class AP：看提升是不是只来自 car/barrier 等大类，还是 bus/truck/pedestrian 等类别也一起提升。
- 梯度和稳定性：是否有 NaN、loss spike、梯度爆炸、学习率 warmup 后突变。
- checkpoint 选择：只看最后 epoch 可能不够，最好比较 best NDS epoch 和 last epoch。
- 重复实验：如果时间允许，同一学习率换 seed 再跑一次，排除单次随机性。

学习率分析依据和补充结论：

数据来源：
- 基准：`lightning_logs/version_6`，`lr_scale=1.0`
- 小学习率：`lightning_logs/version_9`，`lr_scale=0.5`
- 大学习率：`lightning_logs/version_11`，`lr_scale=2.0`
- 训练曲线来自 TensorBoard event 文件，当前代码只记录了 `detection_loss` 和 `depth_loss`。
- 最终检测指标来自三个 last checkpoint 的 256x704 val 测试结果。

代码依据：
- `training_step` 里返回的是 `detection_loss + depth_loss`，但日志只写了 `detection_loss` 和 `depth_loss`。
- `detection_loss` 内部由 head 里的 heatmap focal loss 和 bbox L1 loss 累加而成，但当前代码没有把 heatmap/bbox 子 loss 单独 log 出来。
- `depth_loss` 是前景深度 bin 的 BCE loss，最后乘了 3.0。
- 学习率由 `basic_lr_per_img * batch_size_per_device * gpus * accumulate_grad_batches * lr_scale` 得到，并使用 `MultiStepLR([19, 23])`。

1. loss 曲线：

| exp | detection last100 mean | depth last100 mean | detection last | depth last | 观察 |
|---|---:|---:|---:|---:|---|
| 基准 lr_scale=1.0 | 9.8154 | 7.9757 | 9.4455 | 7.3795 | 正常收敛 |
| 小学习率 lr_scale=0.5 | 10.0585 | 8.0352 | 9.6478 | 7.4431 | 后期 loss 略高于基准 |
| 大学习率 lr_scale=2.0 | 9.6032 | 7.9904 | 8.9256 | 7.3200 | detection loss 后期最低，last loss 也最低 |

结论：
- 大学习率的 detection loss 后期更低，这和它最终 mAP/NDS 更高是匹配的。
- depth loss 三组差异很小，说明学习率变化主要影响检测头/BEV 检测收敛，而不是明显改变 depth supervision 的最终数值。
- 小学习率的早期 detection loss 出现过很大的异常值，但后期恢复正常；从最终指标看，它不是最优。
- 现有日志没有单独的 heatmap loss、bbox/reg loss，所以不能直接说 heatmap 或 bbox 哪个下降更快，只能说 detection loss 总体。

2. validation 指标随 epoch：

当前没有每个 epoch 的 val NDS/mAP 曲线，因为训练阶段没有按 epoch 跑 validation 并记录 NDS；目前只有 last checkpoint 的最终 eval。

所以现在只能比较：

| exp | mAP | NDS |
|---|---:|---:|
| 基准 lr_scale=1.0 | 0.3141 | 0.4158 |
| 小学习率 lr_scale=0.5 | 0.3078 | 0.3955 |
| 大学习率 lr_scale=2.0 | 0.3215 | 0.4275 |

结论：
- 从 last checkpoint 看，`lr_scale=2.0` 最好。
- 但“哪个学习率更早达到高 NDS”“后期是否回落”现在无法证明，需要训练时每个 epoch 都 eval，或者至少保留多个 epoch checkpoint 再逐个测试。

3. 指标拆分：

| exp | mATE | mASE | mAOE | mAVE | mAAE |
|---|---:|---:|---:|---:|---:|
| 基准 lr_scale=1.0 | 0.7322 | 0.2795 | 0.6349 | 0.5500 | 0.2159 |
| 小学习率 lr_scale=0.5 | 0.7187 | 0.2892 | 0.7522 | 0.5999 | 0.2247 |
| 大学习率 lr_scale=2.0 | 0.7050 | 0.2780 | 0.6164 | 0.5071 | 0.2265 |

相对基准：
- 小学习率：mATE 略好，但 mAOE +0.1173、mAVE +0.0499，朝向和速度明显变差，所以 NDS 被拉低。
- 大学习率：mATE -0.0272、mAOE -0.0185、mAVE -0.0429，位置、朝向、速度都更好，所以 NDS 提升。

结论：
- 大学习率不是只提高 AP，它还改善了 localization、orientation、velocity 这些质量指标。
- 小学习率的问题主要不是“完全检不出来”，而是 box 质量，尤其朝向和速度较差。

4. per-class AP：

相对基准，大学习率 AP 变化：

| class | delta AP |
|---|---:|
| car | +0.009 |
| truck | +0.009 |
| bus | +0.017 |
| trailer | +0.023 |
| construction_vehicle | -0.009 |
| pedestrian | +0.002 |
| motorcycle | +0.000 |
| bicycle | +0.016 |
| traffic_cone | +0.001 |
| barrier | +0.004 |

相对基准，小学习率 AP 变化：

| class | delta AP |
|---|---:|
| car | -0.013 |
| truck | -0.007 |
| bus | +0.006 |
| trailer | +0.000 |
| construction_vehicle | -0.007 |
| pedestrian | -0.011 |
| motorcycle | -0.001 |
| bicycle | -0.016 |
| traffic_cone | -0.015 |
| barrier | +0.000 |

结论：
- 大学习率提升不是只来自 car/barrier，大部分主要类别都有小幅提升，bus、trailer、bicycle 更明显。
- construction_vehicle 下降，说明大学习率并不是所有类别都更好；这个类本身样本少、AP 低，波动也更大。
- 小学习率多数类别低于基准，说明它整体没有训练到更好的检测状态。

5. 梯度和稳定性：

已有证据：
- TensorBoard 里的 `detection_loss` 和 `depth_loss` 都是有限值，训练能完整到 epoch 23，并成功保存 ckpt。
- 三个 checkpoint 都能正常 eval，没有 NaN 输出导致 evaluator 崩溃。
- `lr_scale=2.0` 的最大 detection loss 只有 27.99；基准最大 320.13，小学习率最大 4075.75。这个最大值主要出现在最早期记录点，后期都恢复正常。

不能证明的部分：
- 当前没有 grad norm、参数 norm、学习率曲线、NaN counter，所以不能严格判断“是否发生梯度爆炸”。
- 只能说从 loss/event/ckpt/eval 结果看，没有出现训练崩溃或明显不稳定。

6. checkpoint 选择：

当前每组只比较了 `epoch=23-step=10560.ckpt`，也就是 last checkpoint。

结论：
- `lr_scale=2.0` 是 last checkpoint 上最好，不一定等于全训练过程中的 best checkpoint。
- 如果要严谨，应该保存 `every_n_epochs=1` 或按 NDS 监控 best checkpoint，然后比较 best NDS。
- 现在没有每个 epoch 的 ckpt，因此不能排除某个学习率在更早 epoch 曾经更好。

7. 重复实验：

当前每个学习率只有一个 seed，所有结论都是单次实验结论。

结论：
- 可以写成：“在当前 seed 和训练设置下，`lr_scale=2.0` 表现最好。”
- 不建议写成：“大学习率必然优于小学习率。”
- 如果时间允许，至少对基准和 `lr_scale=2.0` 再各跑一个 seed，看 NDS 差距是否仍然稳定。

# 5 depth bin 更细粒度

配置：
- 基准 depth bin：`d_bound=[2.0, 58.0, 0.5]`，depth_channels = 112
- 本实验 depth bin：`d_bound=[2.0, 58.0, 0.25]`，depth_channels = 224
- 图像分辨率和 BEV 网格保持不变：256x704，128x128
- 其他训练设置保持一致：`-b 4 --gpus 1 --accumulate_grad_batches 16`

训练命令：
```
python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025.py \
  -b 4 --gpus 1 --accumulate_grad_batches 16
```

ckpt:
`/home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025/lightning_logs/version_0/checkpoints/epoch=23-step=10560.ckpt`

测试命令：
```
CUDA_VISIBLE_DEVICES=1 conda run -n bevdepth1 python /home/ubuntu/SWW/code/BEVDepth/bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025/lightning_logs/version_0/checkpoints/epoch=23-step=10560.ckpt
```

注意：这次 eval 前发现 `bevdepth/evaluators/det_evaluators.py` 中写 `results_nusc.json` 的地方有一处拼写错误，`self.modalvcity` 已修成 `self.modality`，否则会在 test_epoch_end 写 json 时失败。

结果：
```
mAP: 0.3180
mATE: 0.7132
mASE: 0.2820
mAOE: 0.6112
mAVE: 0.5035
mAAE: 0.2298
NDS: 0.4250
Eval time: 38.4s
```

Per-class results:

Object Class            AP      ATE     ASE     AOE     AVE     AAE
car                     0.491   0.552   0.166   0.202   0.539   0.231
truck                   0.259   0.729   0.227   0.241   0.455   0.212
bus                     0.354   0.766   0.223   0.193   1.017   0.277
trailer                 0.154   1.044   0.230   0.621   0.431   0.201
construction_vehicle    0.062   1.035   0.531   1.415   0.127   0.417
pedestrian              0.281   0.754   0.298   0.974   0.560   0.290
motorcycle              0.324   0.662   0.254   0.834   0.679   0.204
bicycle                 0.309   0.547   0.261   0.801   0.220   0.006
traffic_cone            0.451   0.518   0.342   nan     nan     nan
barrier                 0.496   0.525   0.287   0.219   nan     nan

测试耗时：
- total time = 474.73s
- test_step = 0.16456s/batch ≈ 41.1ms/sample
- batch_to_device = 0.06251s/batch
- test_epoch_end = 116.4s

和基准 #1 对比：

| exp | depth step | depth bins | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | test_step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 基准 | 0.5 | 112 | 0.3141 | 0.4158 | 0.7322 | 0.2795 | 0.6349 | 0.5500 | 0.2159 | 0.15536s/batch |
| 细 depth bin | 0.25 | 224 | 0.3180 | 0.4250 | 0.7132 | 0.2820 | 0.6112 | 0.5035 | 0.2298 | 0.16456s/batch |

变化：
- mAP +0.0039，NDS +0.0092，整体小幅提升。
- mATE、mAOE、mAVE 都变好，说明位置、朝向、速度质量有提升。
- mASE 和 mAAE 略差，尺寸和属性没有同步提升。
- test_step 从 0.15536s/batch 增加到 0.16456s/batch，大约慢 5.9%；depth bin 数量翻倍，但端到端推理没有翻倍，说明耗时不完全由 depth bins 决定。

Per-class AP 相对基准变化：

| class | baseline AP | dbins025 AP | delta |
|---|---:|---:|---:|
| car | 0.485 | 0.491 | +0.006 |
| truck | 0.254 | 0.259 | +0.005 |
| bus | 0.362 | 0.354 | -0.008 |
| trailer | 0.143 | 0.154 | +0.011 |
| construction_vehicle | 0.069 | 0.062 | -0.007 |
| pedestrian | 0.276 | 0.281 | +0.005 |
| motorcycle | 0.315 | 0.324 | +0.009 |
| bicycle | 0.311 | 0.309 | -0.002 |
| traffic_cone | 0.440 | 0.451 | +0.011 |
| barrier | 0.486 | 0.496 | +0.010 |

结论：
- 更细的 depth bin 在当前实验中是有效的，但收益不大，属于小幅提升。
- 提升主要体现在 NDS 相关的 box 质量指标，尤其 mATE/mAOE/mAVE，而不是所有类别 AP 都明显增长。
- 代价是推理略慢、训练也更重；如果后续要继续调，可以再试 `d_bound=[2.0, 58.0, 0.75]` 或只调整近距离深度范围，看看是否存在更划算的精度/速度折中。

# 6 bad case分析

使用工具：
- `visualize/compare_pred_gt.py`：把预测框和 GT 投影到 6 个相机图像，蓝色 TP、红色 FP、黄色 FN。
- `visualize/mine_bad_cases.py`：按 FN/FP、远距离、小目标、低可见度等标签筛 bad case。
- `visualize/dump_bev_depth_outputs.py`：导出输入图、depth argmax/conf、splat 后 BEV、最终 BEV feature、head heatmap，用于辅助判断错误原因。

分析口径：
- 当前 TP/FP/FN 是可视化用的简化匹配，不是 nuScenes 官方 mAP/NDS 的匹配。
- 规则是：先用 `score_thr=0.3` 过滤预测框，再做同类别 BEV 中心距离 greedy matching，`match_dist=2.0m` 内算 TP。
- 未匹配 GT 算 FN，未匹配预测框算 FP。
- 全量统计时只保留 BEVDepth 检测头实际预测的 10 类：`car/truck/bus/trailer/construction_vehicle/pedestrian/motorcycle/bicycle/traffic_cone/barrier`。

## 6.1 全验证集简化统计

基于基准实验 `results_nusc.json`，在 nuScenes val 6019 个 sample 上统计：

| item | count |
|---|---:|
| GT boxes | 187675 |
| Pred boxes(score>=0.3) | 181988 |
| TP | 81006 |
| FN | 106669 |
| FP | 100982 |
| bad samples | 5977 / 6019 |

按 bad case 标签统计：

| tag | count | 含义 |
|---|---:|---|
| `fn_far` | 59367 | 远距离 GT 漏检，距离 >= 40m |
| `fn_small` | 36497 | 小目标 GT 漏检，box volume <= 2 |
| `fn_low_visibility` | 57100 | 低可见度 GT 漏检，visibility token 为 1/2 |
| `fp_far` | 4500 | 远距离误检 |
| `fp_small` | 85683 | 小体积预测框误检 |

按类别统计：

| class | TP | FN | FP | 主要现象 |
|---|---:|---:|---:|---|
| car | 32830 | 47174 | 11041 | 数量最多，远距离和遮挡场景漏检明显 |
| pedestrian | 15952 | 18542 | 46351 | FP 很多，小目标/低分候选多，阈值敏感 |
| traffic_cone | 9415 | 6182 | 22314 | 小目标误检多，容易和路边细长物混淆 |
| barrier | 14930 | 12062 | 16028 | 静态目标较多，近处可检，远处/遮挡仍容易漏 |
| truck | 4008 | 11696 | 2543 | 大车漏检较多，和远距离、遮挡、类别混淆有关 |
| bus | 1133 | 2025 | 341 | FP 少但召回不足 |
| trailer | 674 | 3485 | 492 | 召回弱，类别样本少且形态变化大 |
| construction_vehicle | 199 | 2479 | 235 | AP 本身低，召回非常弱 |
| motorcycle | 990 | 1518 | 952 | 小目标和姿态变化影响较大 |
| bicycle | 875 | 1506 | 685 | 小目标，漏检和定位偏差都存在 |

结论：
- 当前模型的 bad case 主要不是单一问题，而是“远距离 + 小目标 + 低可见度”的叠加。
- FN 中 `fn_far` 和 `fn_low_visibility` 数量都很高，说明相机 BEV 检测对远处、遮挡、截断目标仍然敏感。
- FP 中 `fp_small` 占比极高，主要集中在 pedestrian、traffic_cone、barrier 等小/细长类别，说明低阈值下会产生大量小目标候选。
- construction_vehicle、trailer、truck 这类长尾/大车类召回弱，和前面 per-class AP 较低的现象一致。

## 6.2 单样本阈值敏感性

样本：
`163b70e627854893b88575caf85a56ea`

这个样本一共有 41 个 GT，主要类别是 car、pedestrian、traffic_cone。改变 `score_thr` 后，简化匹配结果变化很明显：

| score_thr | pred | TP | FN | FP | 观察 |
|---:|---:|---:|---:|---:|---|
| 0.7 | 4 | 4 | 37 | 0 | 阈值太高，低置信预测被过滤，画面上大量 GT 变 FN |
| 0.5 | 11 | 8 | 33 | 3 | 召回略升，但仍漏很多小目标/远目标 |
| 0.3 | 24 | 12 | 29 | 12 | 当前可视化默认阈值，召回和误检折中 |
| 0.1 | 164 | 28 | 13 | 136 | 低阈值能找回部分漏检，但 FP 爆炸 |

单样本结论：
- “图上好像有框但仍然是 FN”通常不是代码画错，而是预测框可能被 `score_thr` 过滤、类别不一致，或者 BEV 中心距离超过 `match_dist=2.0m`。
- 降低阈值可以减少 FN，但会显著增加 FP；这个样本从 `score_thr=0.3` 降到 0.1，FN 从 29 降到 13，但 FP 从 12 增到 136。
- 所以 bad case 分析不能只看一张投影图，需要同时看预测分数、类别、BEV 中心距离和 GT 可见度。

## 6.3 可视化观察

已经导出的内部可视化样本：
`outputs/vis_internal/163b70e627854893b88575caf85a56ea/`

包含：
- `input_cam0~5.jpg`：模型实际输入图，已经反归一化，尺寸是 704x256，不是 nuScenes 原图 1600x900。
- `depth_argmax_cam0~5.jpg`：每个相机低分辨率 depth bin 最大响应。
- `depth_conf_cam0~5.jpg`：depth 最大概率，用于观察深度预测是否自信。
- `splat_sweep0_mean/max.jpg`：key frame splat 到 BEV 后的特征响应。
- `splat_sweep1_mean/max.jpg`：第二个 sweep splat 到 BEV 后的特征响应。
- `bev_feature_mean.jpg`：拼接多 sweep 后的最终 BEV feature 响应。
- `head_heatmap_task*.jpg`：检测头不同 task 的 heatmap。

观察结论：
- 输入图被 resize/crop/normalize 后，视觉上和原图不同是正常的；当前保存的是反归一化后的模型输入，不是原始相机图片。
- depth 图是 44x16 的低分辨率特征图，可用于看粗粒度深度趋势，但不能直接当成精确深度图。
- splat 后 BEV 图可以看哪些区域有图像特征被投到 BEV 网格；如果 GT 所在 BEV 区域响应很弱，可能对应深度估计不准、远距离目标特征弱或遮挡严重。
- head heatmap 可以辅助区分两类问题：BEV feature 已有响应但 head 没激活，偏检测头/分类问题；BEV feature 本身弱，偏图像特征、深度或几何投影问题。

## 6.4 后续改进方向

- 在 bad case JSON 里增加 `nearest_pred_score`、`nearest_pred_dist`、`nearest_pred_class`，把“纯漏检”和“近距离但没匹配上”的 near-miss 分开。
- 对 FN 再细分为：远距离漏检、小目标漏检、低可见度漏检、类别错分、定位偏差超过 2m。
- 对 FP 再细分为：低分重复框、小目标误检、类别混淆、远距离背景误检。
- 针对小目标 FP 多的问题，可以尝试调 `score_thr`、NMS/post-processing 阈值，或者按类别设置不同阈值。
- 针对远距离 FN 多的问题，可以结合 depth bin、输入分辨率、远距离数据增强、BEV 网格范围/分辨率继续实验。

# 7 depth预测与点云深度对比

目的：
- 检查模型预测的 depth 分布和 LiDAR 投影得到的真实深度监督是否一致。
- 分析 depth 错误主要出现在近距离、远距离、某些相机，还是概率分布本身不稳定。

分析方法：
- 新增脚本：`visualize/analyze_depth_vs_lidar.py`
- 输入：训练好的 checkpoint、exp 配置、nuScenes val 数据。
- 真实深度：使用数据集里由 LiDAR 点云投影到相机图像得到的 `depth_labels`。
- 对齐方式：复用训练代码里的 `get_downsampled_gt_depth()`，把点云深度下采样到模型 depth 输出的 16x44 特征图上。
- 只在有 LiDAR 深度监督的前景像素上统计误差；没有点云落点的像素不参与统计。

注意：
- 这里比较的是“LiDAR 稀疏点投影深度”，不是全像素稠密深度。
- depth 输出是离散 depth bin 分类，不是直接回归米制深度。
- 基准模型 depth bin 为 `d_bound=[2.0, 58.0, 0.5]`，共 112 个 bin。
- 预测深度用 `argmax depth bin` 转成米制深度；同时也统计了 `expected depth`，即按概率分布求期望。

## 7.1 单样本观察

样本：
`163b70e627854893b88575caf85a56ea`

输出目录：
`outputs/depth_analysis_sample/163b70e627854893b88575caf85a56ea/`

包含：
- `*_input.jpg`：模型输入图，已反归一化。
- `*_gt_depth.jpg`：LiDAR 投影并下采样后的 GT depth。
- `*_pred_argmax_depth.jpg`：模型 argmax depth bin 对应的预测深度。
- `*_pred_expected_depth.jpg`：模型 depth 概率期望。
- `*_depth_conf.jpg`：最大 depth 概率。
- `*_argmax_error.jpg`：argmax depth 与 GT depth 的绝对误差。
- `*_expected_error.jpg`：expected depth 与 GT depth 的绝对误差。

单样本整体结果：

| metric | value |
|---|---:|
| valid LiDAR depth points | 3908 |
| argmax depth MAE | 1.8796m |
| expected depth MAE | 1.9608m |
| median error | 0.5m |
| p75 error | 1.0m |
| p90 error | 5.0m |
| within 1m | 75.7% |
| within 2m | 82.5% |
| within 4m | 88.7% |
| mean depth confidence | 0.382 |

按相机：

| camera | valid points | argmax MAE | expected MAE | within 1m | within 2m | within 4m |
|---|---:|---:|---:|---:|---:|---:|
| CAM_FRONT_LEFT | 664 | 1.00 | 1.15 | 82.7% | 88.3% | 95.0% |
| CAM_FRONT | 601 | 3.31 | 3.74 | 70.7% | 77.9% | 83.9% |
| CAM_FRONT_RIGHT | 666 | 1.99 | 1.87 | 81.1% | 85.1% | 88.0% |
| CAM_BACK_LEFT | 700 | 1.23 | 1.26 | 78.6% | 84.0% | 92.3% |
| CAM_BACK | 578 | 2.48 | 2.34 | 68.3% | 78.2% | 84.6% |
| CAM_BACK_RIGHT | 699 | 1.54 | 1.68 | 71.4% | 80.4% | 87.4% |

单样本结论：
- 大部分有 LiDAR 监督的点误差不大，median 是 0.5m，说明模型在近处/清晰区域能学到有效深度。
- 误差有长尾，p90 到 5m；CAM_FRONT 和 CAM_BACK 明显比左右相机更差。
- `argmax depth` 在这个样本上略优于 `expected depth`，说明概率分布虽然有信息，但用期望会被长尾概率拉偏。

## 7.2 100个val样本统计

命令：
```
CUDA_VISIBLE_DEVICES=0 conda run -n bevdepth1 python visualize/analyze_depth_vs_lidar.py \
  --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6/checkpoints/epoch=23-step=10560.ckpt \
  --max-samples 100 \
  --device cuda:0 \
  --out-dir outputs/depth_analysis_100
```

结果文件：
`outputs/depth_analysis_100/depth_vs_lidar_summary.json`

整体：

| metric | value |
|---|---:|
| mean sample argmax MAE | 2.216m |
| mean sample expected MAE | 2.168m |

按相机统计：

| camera | valid points | argmax MAE | signed error | mean conf | within 1m | within 2m | within 4m |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM_FRONT_LEFT | 66226 | 2.12 | -0.60 | 0.313 | 66.9% | 77.2% | 86.3% |
| CAM_FRONT | 61710 | 2.64 | -0.41 | 0.357 | 66.3% | 74.4% | 83.2% |
| CAM_FRONT_RIGHT | 63732 | 2.59 | -1.24 | 0.323 | 64.8% | 74.3% | 84.2% |
| CAM_BACK_LEFT | 69030 | 1.71 | -0.53 | 0.329 | 70.5% | 81.1% | 89.6% |
| CAM_BACK | 57231 | 2.52 | -0.81 | 0.361 | 66.8% | 75.5% | 83.8% |
| CAM_BACK_RIGHT | 66402 | 1.79 | -0.72 | 0.337 | 71.2% | 81.9% | 89.8% |

说明：
- signed error = `pred_depth - gt_depth`。
- signed error 多数为负，说明模型整体倾向于把深度预测得更近。
- CAM_FRONT、CAM_FRONT_RIGHT、CAM_BACK 的误差更大；CAM_BACK_LEFT、CAM_BACK_RIGHT 相对更好。

按真实深度距离段：

| GT depth range | valid points | argmax MAE | signed error | within 1m | within 2m | within 4m |
|---|---:|---:|---:|---:|---:|---:|
| 2-10m | 222842 | 0.56 | +0.11 | 90.0% | 96.0% | 98.8% |
| 10-20m | 97569 | 2.19 | -0.11 | 52.7% | 70.6% | 86.9% |
| 20-40m | 47197 | 6.32 | -2.12 | 15.4% | 27.4% | 47.5% |
| 40-58m | 16723 | 12.67 | -11.25 | 7.6% | 13.6% | 25.0% |

距离段结论：
- 2-10m 很准，MAE 只有 0.56m，90% 点在 1m 内。
- 10-20m 开始明显变差，MAE 到 2.19m。
- 20m 以后是主要问题：20-40m 的 MAE 6.32m，40-58m 的 MAE 12.67m。
- 远距离 signed error 强烈为负，尤其 40-58m 为 -11.25m，说明远距离点常被预测到更近的 depth bin。
- 这和 bad case 里远距离 FN 多是一致的：远距离深度被拉近，会导致 lift 到 BEV 后的位置偏前，影响检测头召回和定位。

## 7.3 细depth bin对比

对比模型：
- 基准：`d_bound=[2.0, 58.0, 0.5]`，depth bins = 112
- 细 bin：`d_bound=[2.0, 58.0, 0.25]`，depth bins = 224

细 bin 命令：
```
CUDA_VISIBLE_DEVICES=0 conda run -n bevdepth1 python visualize/analyze_depth_vs_lidar.py \
  --exp bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025 \
  --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025/lightning_logs/version_0/checkpoints/epoch=23-step=10560.ckpt \
  --max-samples 100 \
  --device cuda:0 \
  --out-dir outputs/depth_analysis_100_dbins025
```

整体对比：

| exp | depth bins | mean argmax MAE | mean expected MAE |
|---|---:|---:|---:|
| baseline | 112 | 2.216m | 2.168m |
| dbins025 | 224 | 2.175m | 2.122m |

按距离段对比：

| GT depth range | baseline MAE | dbins025 MAE | 变化 |
|---|---:|---:|---:|
| 2-10m | 0.562 | 0.570 | +0.008 |
| 10-20m | 2.195 | 2.139 | -0.056 |
| 20-40m | 6.324 | 6.038 | -0.286 |
| 40-58m | 12.666 | 12.730 | +0.064 |

结论：
- 细 depth bin 对整体深度误差有小幅改善，mean argmax MAE 从 2.216m 降到 2.175m。
- 改善主要来自 10-40m，尤其 20-40m 从 6.32m 降到 6.04m。
- 2-10m 基本不变，40m 以上也没有改善。
- 这和 #5 的检测结果一致：细 bin 带来小幅 NDS 提升，但不是根治远距离深度错误。

## 7.4 问题定位

主要问题：
- 远距离深度低估：20m 以后误差快速变大，40m 以上平均偏近 11m 左右。
- 中心前/后视角误差偏大：CAM_FRONT、CAM_FRONT_RIGHT、CAM_BACK 比部分侧后视角更差。
- depth 监督稀疏：统计只来自 LiDAR 投影点，很多图像区域没有直接深度监督；远距离点更稀疏，监督更弱。
- 下采样带来的遮挡/混合问题：训练时一个 16x 下采样 cell 里取最近深度点，远处目标和近处遮挡物容易混在同一个 cell，模型会偏向近处深度。
- argmax/expected 差异不大：说明问题不是单纯后处理取 argmax 造成，而是 depth 分布本身在远距离区域就不够准。

后续可做：
- 对远距离区域单独可视化 GT depth / pred depth / error，确认是否集中在路边小目标、遮挡行人、远处车辆。
- 加 near/far 分段 depth loss 或者远距离点加权，缓解远处监督被近处点主导。
- 试非均匀 depth bins：近处细、远处也保留足够分辨率，而不是全范围均匀细化。
- 结合 bad case：把远距离 FN 的 sample token 和 depth error 图对应起来，看漏检是否发生在 depth 被明显预测近的位置。

## 7.5 远距离FN和depth error对应

目的：
- 把远距离 FN 的 `sample_token / gt_token` 和局部 depth error 对起来。
- 看漏检是否经常发生在“GT 所在图像区域被预测得明显更近”的位置。

新增脚本：
- `visualize/correlate_far_fn_depth.py`

分析方法：
1. 读取 baseline `results_nusc.json`。
2. 用 `score_thr=0.3`、`match_dist=2.0m` 做简化匹配，筛选远距离 FN。
3. 只看 `40m <= ego_distance <= 58m` 的 FN，因为 baseline depth head 的 `d_bound=[2.0, 58.0, 0.5]`，超过 58m 已经不在 depth 可预测范围内。
4. 把 FN GT box 投影到相机图像，并转换到模型输入图 704x256 和 depth 特征图 44x16 上。
5. 在 GT box 覆盖的 depth 特征区域里统计 `pred_depth - lidar_depth`。
6. 如果局部 signed error 小于 -2m，认为这个 FN 区域存在“明显预测近”的现象。

命令：
```
CUDA_VISIBLE_DEVICES=0 conda run -n bevdepth1 python visualize/correlate_far_fn_depth.py \
  --ckpt outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6/checkpoints/epoch=23-step=10560.ckpt \
  --pred outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/results_nusc.json \
  --score-thr 0.3 \
  --match-dist 2.0 \
  --far-dist 40 \
  --max-dist 58 \
  --max-cases 50 \
  --device cuda:0 \
  --out outputs/far_fn_depth_correlation_40_58.json
```

结果文件：
- `outputs/far_fn_depth_correlation_40_58.json`
- 对比图：`outputs/far_fn_depth_examples/contact_sheet.jpg`

整体统计：

| item | value |
|---|---:|
| analyzed far FN cases | 50 |
| cases with LiDAR depth in projected region | 48 |
| predicted-nearer cases signed error < -2m | 16 |
| predicted-nearer ratio | 33.3% |
| mean local signed error | +0.47m |
| mean local MAE | 6.83m |

解释：
- 不是所有远距离 FN 都是“深度预测近”导致的。
- 在 40-58m 远距离 FN 中，大约 1/3 的 GT box 局部区域确实被预测得明显更近。
- 但也有不少反例是预测偏远，或者局部 LiDAR 点很少，说明远距离 FN 还混有分类置信度低、目标过小、遮挡、可见度低、检测头未激活等原因。
- `mean local signed error` 为正，说明这 50 个最远 FN 的局部误差方向是混合的；不能简单归因成“远距离全部预测近”。

典型“预测近”的例子：

| sample_token | gt_token | class | dist | camera | local signed error | local MAE | 说明 |
|---|---|---|---:|---|---:|---:|---|
| `83d3ee0e085b4ac282e06741fe1f3ae2` | `ba8245e9c3004389aa69c2162234acf4` | car | 58.0m | CAM_BACK_LEFT | -14.90m | 14.90m | GT 区域深度被明显预测近 |
| `9699d6a8d9384f8885e8c5318bc621ab` | `34335b7b4af349308193c8cd6c452ae6` | barrier | 58.0m | CAM_FRONT | -17.10m | 17.90m | GT 区域误差大且方向偏近 |
| `8f1dfb1a348a42f4b9c9934da7492a6e` | `54745d4c671a477ea037ac0f20cbe92c` | truck | 58.0m | CAM_BACK | -10.07m | 10.36m | 远距离大车被投到更近 depth |

典型反例：

| sample_token | gt_token | class | dist | local signed error | local MAE | 说明 |
|---|---|---|---:|---:|---:|---|
| `d63d4b75524f4c77aa9c7f070b006911` | `2f292a345bfa4bf9a1933a330e710ab2` | barrier | 58.0m | +17.50m | 17.50m | 不是预测近，而是预测远 |
| `8cd9b9f28b6b44e3933216e65bbfbbd4` | `b20cf436391f49269d347552ffc4c971` | truck | 58.0m | +13.75m | 13.75m | 远距离 FN 但 depth 方向不是偏近 |

对比图说明：
- `outputs/far_fn_depth_examples/contact_sheet.jpg` 中每个样本包含 input、GT depth、pred depth、argmax error。
- 黄色框是远距离 FN GT box 投影区域。
- 前 3 个样本是局部 signed error 明显为负，也就是预测深度比 LiDAR 深度更近。
- 后 2 个样本是反例，说明 FN 不一定由预测近造成。

结论：
- “远距离 depth 被预测近”确实是远距离 FN 的一个重要原因，但不是唯一原因。
- 在 40-58m 可预测范围内，约 1/3 远距离 FN 有明显预测近现象。
- 更严谨的 bad case 分类可以把远距离 FN 分成：
  - `far_fn_depth_too_near`
  - `far_fn_depth_too_far`
  - `far_fn_no_lidar_depth_in_box`
  - `far_fn_low_score_or_no_heatmap`
- 下一步如果要继续深入，应该把 head heatmap 一起叠到这些 FN box 区域上，看“depth 已经错了”还是“depth 有响应但检测头没激活”。
