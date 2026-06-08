# 环境配置
直接复用了之前的mm环境，然后就出现了问题具体：
-  我改了requirements.txt ，已经被改成了 Torch 2.5.1 / Lightning 2.4.0；
- 这类改动会直接影响 mmdet3d、mmcv、Lightning CLI、CUDA extension；
- 所以我先验证 import torch / mmdet3d / mmcv / pytorch_lightning
还是按照原配置来吧
```bash
# 1. 建立
conda create -n bevdepth python=3.9 -y
conda activate bevdepth
# 2. torch有改动
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 关键点：RTX 4090 是 sm_89，但 torch 1.9/cu111 的 arch list 里没有 sm_89。所以虽然 torch.cuda.is_available() 是 True，但跑到某些 CUDA 底层库，比如 cusparseCreate(handle)，就会炸：
pip uninstall -y torch torchvision torchaudio mmcv mmcv-full

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full==1.6.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# 这里不合适这个新的mmdet别看这个
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4' # In MMCV-v2.x, mmcv-full is renamed to mmcv
mim install 'mmdet>=3.0.0'

# use mmdet3d as a dependency or third-party package, install it with MIM
# mmdet https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html
mim install "mmdet3d>=1.1.0"

# 3. 针对这个库一些包
pip uninstall -y mmcv mmcv-full mmdet mmdet3d mmengine

pip install mmcv-full==1.6.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
pip install mmdet3d==1.0.0rc4

# 4. 降低pip和numpy
python -m pip install pip==24.0 numpy==1.23.5 networkx==2.8.8

# 环境验证
python - <<'PY'
import mmcv, mmdet, mmdet3d, torch, pytorch_lightning
print('torch', torch.__version__)
print('mmcv', mmcv.__version__, hasattr(mmcv, 'dump'))
print('mmdet', mmdet.__version__)
print('mmdet3d', mmdet3d.__version__)
print('pl', pytorch_lightning.__version__)
PY
# 5. 安装requriment，又是numpy core的报错，降低numpy，编译
# 这个如果重安装顶层这里就不用在来一遍了 会改变numpy
pip install -r requirements.txt 
python -m pip install numpy==1.23.5

rm -rf build BEVDepth.egg-info
rm -f bevdepth/ops/voxel_pooling_train/voxel_pooling_train_ext*.so
rm -f bevdepth/ops/voxel_pooling_inference/voxel_pooling_inference_ext*.so
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
python setup.py develop
```
‼️注意：cudnn不兼容的问题
import torch
torch.backends.cudnn.enabled = False


# 数据集
```bash
# 绑定
ln -s [nuscenes root] ./data/
# 自查,如果是openpcdet里面生成的，应该不会有这几个文件
ls -lh data/nuScenes/nuscenes_infos_train.pkl \
       data/nuScenes/nuscenes_infos_val.pkl \
       data/nuScenes/nuscenes_infos_test.pkl
# 生成这里需要的处理
python scripts/gen_info.py
# 看看具体传感器的info生成的信息的存储格式
python scripts/nus_data.py
```

info 存储的是：
```py
info['sample_token'] = cur_sample['token']
info['timestamp'] = cur_sample['timestamp']
info['scene_token'] = cur_sample['scene_token']
info['cam_infos'] = cam_infos # sample
info['lidar_infos'] = lidar_infos # sample
info['cam_sweeps'] = cam_sweeps
info['lidar_sweeps'] = lidar_sweeps
info['ann_infos'] = ann_infos
```

# 动手
## 0. 冒烟检查 即test里面的文件
voxel_pooling CUDA op: 通过
nuScenes dataset test: 通过
真实 train pkl 取 batch: 通过
其真实batch形状：
sweep_imgs: (1, 2, 6, 3, 256, 704)
mats: sensor2ego/intrin/ida/... 
depth: (1, 1, 6, 256, 704)
dataset len: 28130

## 1. 先跑非EMA，小batch，单卡，少step
```bash
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -b 1 \
  --gpus 1 \
  --max_epochs 1 \
  --limit_train_batches 10 \
  --limit_val_batches 0
```
来看：
`dataset -> dataloader -> model forward -> loss -> backward`
不要先跑 _ema.py。EMA 会多一层 callback 和权重滑动平均，对熟悉主流程没帮助。等普通训练能跑通后，再看：
`python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key_ema.py...`
## 2. 调试入口
主要看这几个文件就够了：
```
bevdepth/exps/base_cli.py
bevdepth/exps/nuscenes/base_exp.py
bevdepth/datasets/nusc_det_dataset.py
bevdepth/models/base_bev_depth.py # -->model
bevdepth/layers/backbones/base_lss_fpn.py # -->backbone
bevdepth/layers/heads/bev_depth_head.py # -->head
```
主流程是：
```
exp.py
 -> run_cli()
 -> BEVDepthLightningModel
 -> train_dataloader()
 -> NuscDetDataset.__getitem__()
 -> training_step()
 -> BaseBEVDepth.forward()
 -> get_targets/loss
```
## 3. 如果想更适合 debug
可以临时加断点：
```
import pdb; pdb.set_trace()
```
优先放在：
```
bevdepth/exps/nuscenes/base_exp.py:training_step
bevdepth/datasets/nusc_det_dataset.py:__getitem__
```
然后用：
```bash
python -m pdb bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py -b 1 --gpus 1 --max_epochs 1 --limit_train_batches 1 --limit_val_batches 0
```

## 4. 主线
`数据怎么加载 -> batch 长什么样 -> 模型 forward 经过哪些模块 -> loss / log / checkpoint 输出到哪里`
  - 数据加载
  base_exp.py
  读取：
  nusc_det_dataset.py
  - 训练时候batch被拆开
  base_exp.py
  - 模型入口
  base_bev_depth.py
  会forward主线
```
BEVDepthLightningModel.training_step
-> BEVDepthLightningModel.forward
-> BaseBEVDepth.forward
-> BaseLSSFPN.forward
-> BaseLSSFPN._forward_single_sweep
-> get_cam_feats
-> img_backbone: ResNet
-> img_neck: SECONDFPN
-> depth_net
-> get_geometry
-> voxel_pooling_train
-> BEVDepthHead.forward
-> get_targets
-> loss
```
  打印模块树
```py
如果想直接打印模块树，可以在调试控制台里：
print(model.model)
print(model.model.backbone)
print(model.model.head)
```
  - 输出
  base_cli.py
  可以看到在output里面
  可视化训练结果：
  `tensorboard --logdir outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs`

**调试启动方式**
因为现在 cu113 + 4090 需要禁用 cuDNN，调试时最好用一个小 wrapper 或在入口前加：
```
import torch
torch.backends.cudnn.enabled = False
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -b 1 \
  --gpus 1 \
  --precision 32 \
  --max_epochs 1 \
  --limit_train_batches 1 \
  --limit_val_batches 0
```
观察顺序可以是：
先看 NuscDetDataset.__getitem__ 输出单样本
再看 collate_fn 后 batch
再看 training_step 拆 batch
再看 BaseBEVDepth.forward
最后看 head loss 输出


## 5. 说明
在这个仓库里，一个 exp.py 不是单纯配置文件，它是“配置 + 模型类小改动 + 启动入口”的组合。比如：
`bev_depth_lss_r50_256x704_128x128_24e_2key.py`
名字可以拆成：
```
bev_depth       模型/任务
lss             view transformer 路线
r50             ResNet-50 backbone
256x704         输入图像尺寸
128x128         BEV 网格尺寸
24e             训练 24 epochs
2key            当前关键帧 + 额外一帧
```
它最后会调用：
`run_cli(BEVDepthLightningModel, 'exp_name')`

`IDA`
ida = Image Data Augmentation
它描述图像增强对像素坐标的影响，比如：
resize
crop
flip
rotate
原始图像像素点经过这些增强后，坐标会变，所以需要一个 ida_mat 记录：

原始图像像素坐标 -> 增强后图像像素坐标
例如一张图从原始尺寸 resize/crop 到 256x704，那么像素点 (u, v) 的位置变了。模型后面要把图像特征点反投影到 3D，就必须知道这个图像增强矩阵，否则像素坐标会对不上。
`BDA`
bda = BEV Data Augmentation
它描述 BEV / 3D 空间的数据增强，比如：
BEV 平面旋转
BEV 平面缩放
x/y 翻转
也就是对 3D box、点云、BEV 坐标做的增强。比如训练时随机把整个鸟瞰坐标系旋转几度，GT box 也跟着旋转，BEV 几何坐标也要一起变。



## 6. 实际观察

从bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py

可进入bevdepth/models/base_bev_depth.py 和 bevdepth/exps/base_cli.py

从bevdepth/models/base_bev_depth.py：（模型维度）
bevdepth/layers/backbones/base_lss_fpn.py （backbone）
bevdepth/layers/heads/bev_depth_head.py （head）

从bevdepth/exps/base_cli.py：（训练维度）
bevdepth/exps/nuscenes/base_exp.py （传入模型参数，定义class BEVDepthLightningModel，数据加载，loss）

数据类别以及加载：bevdepth/datasets/nusc_det_dataset.py

### 1. 先看batch
在base_exp.py 中 def training_step(self, batch) batch包括关键张量：
batch len为7 和代码对应
(sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
_ 一个是[B,key frame,cam_num] 维度的张量 全数字应该是对应相机照片的timestamps
_ 一个是列表：[{'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>, 'ego2global_translation': array([1366.35670691, 2671.85694007,    0.        ]), 'ego2global_rotation': array([ 0.94076915, -0.00323019,  0.00336038, -0.33901568]), 'token': 'a4d6e99f30ed4c5589bf6f87503fa064'}]（nuScenes sample token）
```
- sweep_imgs:
张量
[B, num_sweeps, num_cams, 3, H, W]
例如 b=4, 2key, 6相机, 256x704:
[4, 2, 6, 3, 256, 704]
- mats:
len为5的字典
一堆坐标变换矩阵
sensor2ego_mats / intrin_mats / ida_mats / sensor2sensor_mats:
[B, num_sweeps, num_cams, 4, 4]
bda_mat
[B, 4, 4]
- gt_boxes:
长度为 B 的 list
每个元素 shape 大概是 [num_gt, 9]
一个盒子：[14.257, 14.009, 0.516, 4.424, 1.969, 1.535, 1.485, 0.247, 4.480]
[x, y, z, dx, dy, dz, yaw, vx, vy]
x, y, z      物体中心点坐标，ego / lidar 坐标系下，单位 m
dx, dy, dz   3D box 尺寸，长宽高，单位 m
yaw          绕 z 轴旋转角，单位 rad
vx, vy       物体速度在 x/y 方向的分量，单位是 m/s
即：
中心点: x=14.26m, y=14.01m, z=0.52m
尺寸:   dx=4.42m, dy=1.97m, dz=1.54m
朝向:   yaw=1.49 rad
速度:   vx=0.25m/s, vy=4.48m/s

- gt_labels:
长度为 B 的 list
每个元素 shape 是 [num_gt,] 的张量
比如b=1 有3个gt [tensor([0, 8, 0], device='cuda:0')]
- depth_labels:
监督深度图，和相机/图像尺度有关
大小为[B,1,cam_num,256,704]的张量 深度只监视当前key frame
输入图像 sweep_imgs: 有 2 个 key frames 是本帧和前一帧[0,-1]
由于use_fusion=False depth_labels只给当前 key frame 生成 1 份深度监督
```
### 2. 模型大模块
>看forward。
>这个模块输入 shape 是什么？
>输出 shape 是什么？
>坐标系有没有变？
>语义有没有变（比如 image feature -> BEV feature）

bevdepth/layers/models/base_bev_depth.py (模型)
forward:
- backbone输出x（多相机融合后的bev_feature）: torch.Size([1, 160, 128, 128]) [B, C, H_bev, W_bev]
- backbone输出depthpred（每个相机图像特征点的深度分布）: torch.Size([6, 112, 16, 44]) [B * num_cams, D, H_img_feat, W_img_feat] D 是 depth bins（depth_channels = (d_bound[1] - d_bound[0]) / d_bound[2]获得的）16 和 44 是从 256 704 下采样16倍得到的。
- 经过head的preds 是tuple(list[dict]) : ([{...}], [{...}], [{...}], [{...}], [{...}], [{...}]) 6个list，对应6个task，类似：
```
task 0: car, truck
task 1: construction_vehicle, bus
task 2: trailer, barrier
task 3: motorcycle, bicycle
task 4: pedestrian, traffic_cone
...
```
每个list有1个dict 每个dict有 6个量
[batch, channels, bev_h, bev_w]
0. reg torch.Size([1, 2, 128, 128])-->BEV grid中心点 xy 偏移
1. height [1 1 128 128]-->目标中心点高 z
2. dim [1 3 128 128]-->box 长宽高
3. rot [1 2 128 128]-->yaw 编码 用2个通道表示朝向避免角度直接回归不连续
4. vel [1 2 128 128]-->平面速度 vx vy
5. heatmap [1 2 128 128]-->每类目标中心概率图c=2的原因是我看的这个task只有2个类别。


#### 深入模块
bevdepth/layers/backbones/base_lss_fpn.py （网络）
- BaseLSSFPN 类（整个backbone） 从 forward 到 _forward_single_sweep（`get_cam_feats` -> `_forward_depth_net` -> `get_geometry` -> `相乘lift` -> `_forward_voxel_net`->`voxel_pooling_train`）获得feature map。get_cam_feats从sweep_imgs获得图像特征，_forward_depth_net具体看depthnet处理，get_geometry是分配坐标，_forward_voxel_net是选做要不要平滑, voxel_pooling_train是bev特征映射就是splat。

- get_cam_feats函数: 图像特征被flatten得到torch.Size([6, 3, 256, 704])，这个6是batch_size * num_sweeps * num_cams,这个后经过 img_backbone（resnet50）和 img_neck（SECONDFPN）得到图像特征 torch.Size([6, 512, 16, 44])(cam_num,c,h,w) 返回还原成[1, 1, 6, 512, 16, 44]（图像特征）。

- 特色模块DepthNet类：ASPP，MLP，SElayers。此时输入是图像特征，return torch.cat([depth, context], dim=1)。 相机的内参、增强策略、外参一起被mlp后和图像特征分别得到context和depth特征，具体处理时候depth分支会有膨胀卷机扩大感受野。这里输出shape = torch.Size([batch_size * num_cams, 192, 16, 44]) 其中depth:[6, 112, 16, 44] context:[6, 80, 16, 44]

然后把depth部分（截取前112维度）变成深度分布概率，112 个 depth bin 的概率和 = 1（它不是直接预测一个深度值，而是预测这个图像点落在 2m、2.5m、3m、...、57.5m 的概率）。
- get_geometry部分: 给每个[1, 6, 112, 16, 44]的点算一个概率坐标。第 n 个相机图像上，第 h,w 个特征点，如果深度是第 d 个 bin，它在 ego/BEV 空间里对应哪个 3D 点。这里输出torch.Size([1, 6, 112, 16, 44, 3])。接着将真实米映射到bev grid坐标系大小下，这个 frustum 点应该落到 BEV 网格的哪个格子里。

- lift,depth和context相乘：之前的的depth经过和context（depth:[6, 112, 16, 44] context:[6, 80, 16, 44]）得到img_feat_with_depth，其shape = torch.Size([6, 80, 112, 16, 44])，以上是2D image feature -> 3D frustum feature lift。

- _forward_voxel_net部分:如果当前 use_da=False，所以它基本啥也没做，直接返回。如果 use_da=True，它会用 DepthAggregation 在这个特征上再做一层特征聚合，详见DepthAggregation类别，让深度维和图像空间附近的特征更平滑、更有上下文。这里输出是shape = torch.Size([6, 80, 112, 16, 44])。然后再reshape成torch.Size([1, 6, 80, 112, 16, 44])，重排成torch.Size([1, 6, 112, 16, 44, 80])，使得每个空间点携带 80 维特征。-->也就是论文中的DR模块。

- 变成bev的关键:voxel_pooling_train 函数，这部分输入geom_xyz: [1, 6, 112, 16, 44, 3]，img_feat_with_depth:[1, 6, 112, 16, 44, 80]，voxel_num: [Xnum, Ynum, Znum]。遍历所有 camera/depth/h/w 的 frustum points，根据 geom_xyz 找到 BEV 网格位置，把对应的80维 feature 累加/池化到 BEV cell输出shape = 这部分 torch.Size([1, 80, 128, 128])。就是splat。
```text
整个过程总结：
图像特征 [6,512,16,44]
  -> DepthNet 分成 depth概率 [6,112,16,44] 和 context特征 [6,80,16,44]
  -> depth概率 * context特征，得到 frustum feature [6,80,112,16,44]
  -> get_geometry 计算每个 frustum feature 对应的 BEV 网格坐标
  -> voxel_pooling 把所有相机/深度/像素点的特征累加到 BEV
  -> 得到 BEV feature [1,80,128,128]
```


### 3. loss查看
bev_depth_head.loss
1. heatmap focal loss: loss_heatmap -> self.loss_class GaussianFocalLoss(centerpoint head)
2. Regression loss for dimension, offset, height, rotation (reg dim vel) ->self.loss_bbox L1Loss(centerpoint head)

### 4. 检测头结构查看
1. 整体结构 bev_backbone -> bev_neck -> out
2. bev_backbone: resnet18 bev_neck: SECONDFPN 

关键张量表
stage                 tensor          shape                         meaning
dataset               sweep_imgs      [B,S,N,3,H,W]                 多帧多相机图像
LSS image encoder     img_feat        [B*S*N,C,h,w]                 图像特征
depth net             depth_pred      [B*N,D,h,w]                   深度分布
voxel pooling         bev_feat        [B,C_bev,Y,X]                 BEV特征
head                  preds           list[task][dict]              检测头输出
loss                  detection_loss  scalar                        3D检测损失
loss                  depth_loss      scalar                        深度监督损失







## 7. 学习率计算
basic_lr_per_img = 2e-4 / 64
global_batch = self.batch_size_per_device * self.gpus
lr = basic_lr_per_img * global_batch
batch越小，噪声梯度越大，因此，学习率得变小，训练更稳定。
```python
    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]
```
我的解决： 用 gradient accumulation 模拟 global batch 64，如果单卡 且 -b=1，则--accumulate_grad_batches 64 （accumulate_grad_batches 是 PyTorch Lightning 的参数）,如此组合：
```
-b 1  accumulate 64
-b 2  accumulate 32
-b 4  accumulate 16
-b 8  accumulate 8
```
目前尝试了 4 16组合是OK的。提速明显。
self.log('detection_loss', detection_loss, prog_bar=True) 这个可以看loss了，开始时候detection loss非常大，是e+4量级。
注意：如果 -b 1 --accumulate_grad_batches 64，每个 epoch 还是会 forward/backward 28130 次，只是 64 次才 step 一次 ， 更新次数变少了。

## 8. 看loss
tensorboard \
  --logdir /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6 \
  --host 0.0.0.0 \
  --port 6006


# 命令大全
训练
```
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py   -b 4 --gpus 1 --precision 32   --max_epochs 1 --limit_train_batches 20 --limit_val_batches 0
```

测试
```
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -e \
  -b 4 \
  --gpus 1 \
  --precision 32 \
  --ckpt_path /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6/checkpoints/epoch=23-step=10560.ckpt
```

训练查看
```
tensorboard \
  --logdir /home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6 \
  --host 0.0.0.0 \
  --port 6006
```

resolution的批量测试
```
CKPT=/home/ubuntu/SWW/code/BEVDepth/outputs/bev_depth_lss_r50_256x704_128x128_24e_2key/lightning_logs/version_6/checkpoints/epoch=23-step=10560.ckpt
# 测试 192x640：
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_192x640_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path $CKPT
# 测试原始 256x704：
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path $CKPT
# 测试 320x864：
python bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_320x864_128x128_24e_2key.py \
  -e -b 4 --gpus 1 --precision 32 \
  --ckpt_path $CKPT

```


# 一些gpu监看
## 训练开始
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Global seed set to 0
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]

  | Name  | Type         | Params
---------------------------------------
0 | model | BaseBEVDepth | 76.4 M
---------------------------------------
76.4 M    Trainable params
9.5 K     Non-trainable params
76.4 M    Total params
305.538   Total estimated model params size (MB)

解读：76.4 M 个 Trainable params 大小 305.538MB
然而，一个ckpt要：模型参数 305 MB 梯度/优化器状态 600 MB+ 其他状态 若干
合计接近 877 MB 。

## 训练结束
pytorch-lightning 的profiler 
```
Total: 76.936s
train_dataloader: 33.565s  43.6%
run_training_batch: 27.424s 35.6%
training_step: 15.749s 20.5%
backward: 8.121s 10.6%
batch_to_device: 1.077s 1.4%
```
train_dataloader 33.565s
这是构建/初始化 dataloader 的时间，只调用 1 次，但占了 43.6%。所以你这次只跑 20 个 batch 时（每个epoch只跑20个train batch，实际训练了80个样本），它显得特别大。完整 epoch 里这部分会被摊薄一些。
run_training_batch 1.371s / batch
这是每个 batch 的整体训练时间。batch size = 4，样本数 = len(train_dataset)=28130，每 epoch iter 数 ≈ ceil(样本数 / batch_size)=7033，7033 iter × 1.371s ≈ 9642s ≈ 2.68h，我实际是1.5h，前面慢可能是前几个 batch 有 warmup、缓存未命中、保存 checkpoint、初始化等影响。

training_step 0.787s / batch
模型 forward + loss 主要在这里。cuDNN 能影响其中一部分卷积，但不是全部。
backward 0.406s / batch
反向传播时间。这个不算离谱。
batch_to_device 0.054s / batch
数据搬到 GPU 很少，不是瓶颈。

# 一些结论
- 从论文看结果，这几个重要模块的提点排序如下（我们使用的方法是不带dr的）：
Depth Loss：有
Camera-awareness：有
multi-frame：有
Depth Refinement Module：无
- 论文中由于深度预测不准，关于图像大小可能带来的过拟合问题：只验证了小图像测大图像，我的想法是在这里我再训练一个大图像的，测小中图像，看看哪个鲁棒性更好。
- 这个项目有并行训练，batch给很大，我没这个条件，我用了累计法，那就可以调节累积步数改变学习率。时间上我的4090也是和论文8卡一样用了1.5天。
- 深度分桶论文没测，我可以跑一跑看看。

# spec
相机顺序：
0 CAM_FRONT_LEFT
1 CAM_FRONT
2 CAM_FRONT_RIGHT
3 CAM_BACK_LEFT
4 CAM_BACK
5 CAM_BACK_RIGHT

结果生成：
在/home/ubuntu/SWW/code/BEVDepth/bevdepth/evaluators/det_evaluators.py
```json
{
  "meta": {
    "use_lidar": false,
    "use_camera": true,
    "use_radar": false,
    "use_map": false,
    "use_external": false
  },
  "results": {
    "sample_token": [
      {
        "sample_token": "...",
        "translation": [x, y, z],
        "size": [w, l, h],
        "rotation": [qw, qx, qy, qz],
        "velocity": [vx, vy],
        "detection_name": "car",
        "detection_score": 0.83,
        "attribute_name": "vehicle.parked"
      }
    ]
  }
}
```
