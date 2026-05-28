#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=bevdepth1
REPO_DIR=/home/ubuntu/SWW/code/BEVDepth
CONDA_BASE=/home/ubuntu/anaconda3
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"

cd "${REPO_DIR}"
export CUDA_HOME="${ENV_DIR}"
export CUDA_PATH="${ENV_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

echo "==> Environment"
conda run -n "${ENV_NAME}" python -V
"${ENV_DIR}/bin/nvcc" --version

echo "==> Install PyTorch 1.13.1 with CUDA 11.7"
conda install -n "${ENV_NAME}" -y -c pytorch -c nvidia \
  pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7

echo "==> Pin pip for old pytorch-lightning metadata"
"${ENV_DIR}/bin/python" -m pip install pip==24.0

echo "==> Install OpenMMLab stack"
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir \
  mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

echo "==> Install common Python dependencies"
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir \
  numpy==1.23.5 \
  networkx==2.8.8 \
  setuptools==59.5.0 \
  wheel \
  tqdm \
  packaging \
  PyYAML \
  protobuf==3.20.3 \
  typing-extensions \
  pyDeprecate==0.3.2 \
  torchmetrics==0.11.4 \
  fsspec==2023.12.2 \
  tensorboard==2.14.0 \
  tensorboardX \
  nuscenes-devkit \
  opencv-python-headless==4.8.1.78 \
  pandas \
  scikit-image \
  scipy \
  numba \
  pycocotools \
  terminaltables \
  plyfile \
  trimesh==2.35.39 \
  mmcls==0.25.0 \
  prettytable \
  pytest

echo "==> Install Lyft SDK without letting it upgrade numpy"
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir --no-deps \
  lyft-dataset-sdk==0.0.8

echo "==> Install Lightning without resolver backtracking"
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir --no-deps \
  pytorch-lightning==1.6.0

echo "==> Install OpenMMLab Python packages without resolver backtracking"
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir --no-deps \
  mmdet==2.28.2 \
  mmsegmentation==0.30.0 \
  mmdet3d==1.0.0rc4

echo "==> Pin compatibility packages"
"${ENV_DIR}/bin/python" -m pip uninstall -y opencv-python || true
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir \
  numpy==1.23.5 \
  networkx==2.8.8 \
  opencv-python-headless==4.8.1.78 \
  pip==24.0

echo "==> Store CUDA env vars in conda env"
conda env config vars set -n "${ENV_NAME}" \
  CUDA_HOME="${ENV_DIR}" \
  CUDA_PATH="${ENV_DIR}"

echo "==> Build BEVDepth extensions"

rm -rf build BEVDepth.egg-info
rm -f bevdepth/ops/voxel_pooling_train/voxel_pooling_train_ext*.so
rm -f bevdepth/ops/voxel_pooling_inference/voxel_pooling_inference_ext*.so

"${ENV_DIR}/bin/python" setup.py develop

echo "==> Verify imports"
"${ENV_DIR}/bin/python" - <<'PY'
import torch
import mmcv
import mmdet
import mmdet3d
import pytorch_lightning as pl
import numpy
import networkx

print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
print("mmdet3d", mmdet3d.__version__)
print("pytorch_lightning", pl.__version__)
print("numpy", numpy.__version__)
print("networkx", networkx.__version__)
PY

echo "==> Run voxel pooling test"
"${ENV_DIR}/bin/python" -m pytest test/test_ops/test_voxel_pooling.py -q

echo "==> Done"
