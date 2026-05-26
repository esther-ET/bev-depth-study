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
# 2. torch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 这里不合适这个新的mmdet
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
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
pip install mmdet3d==1.0.0rc4

# 4. 降低pip和numpy
conda run -n bevdepth python -m pip install pip==24.0
python -m pip install numpy==1.23.5
# 环境验证
python - <<'PY'
import mmcv, mmdet, mmdet3d, torch, pytorch_lightning
print('torch', torch.__version__)
print('mmcv', mmcv.__version__, hasattr(mmcv, 'dump'))
print('mmdet', mmdet.__version__)
print('mmdet3d', mmdet3d.__version__)
print('pl', pytorch_lightning.__version__)
PY
# 5. 安装requriment，降低numpy，编译
pip install -r requirements.txt
python -m pip install numpy==1.23.5
python setup.py develop
```


# 数据集合
```bash
# 绑定
ln -s [nuscenes root] ./data/
# 自查,如果是openpcdet里面生成的，应该不会有这几个文件
ls -lh data/nuScenes/nuscenes_infos_train.pkl \
       data/nuScenes/nuscenes_infos_val.pkl \
       data/nuScenes/nuscenes_infos_test.pkl
# 生成这里需要的处理
python scripts/gen_info.py
```