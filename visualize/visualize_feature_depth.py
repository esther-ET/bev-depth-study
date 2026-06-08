import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize saved BEV feature, depth logits, or heatmap tensors.')
    parser.add_argument('--tensor', required=True, help='.npy or torch .pt/.pth')
    parser.add_argument('--kind',
                        choices=['bev', 'depth_argmax', 'depth_conf', 'heatmap'],
                        required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--camera-index',
                        type=int,
                        default=0,
                        help='Used when tensor contains camera dimension.')
    return parser.parse_args()


def load_tensor(path):
    path = Path(path)
    if path.suffix == '.npy':
        arr = np.load(path)
    else:
        import torch
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, dict) and 'tensor' in obj:
            obj = obj['tensor']
        if hasattr(obj, 'detach'):
            obj = obj.detach().cpu()
        arr = np.asarray(obj)
    return arr


def normalize_to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr)
    if denom > 0:
        arr = arr / denom
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def colorize(gray):
    gray = normalize_to_uint8(gray)
    try:
        import matplotlib.cm as cm
        colored = cm.get_cmap('magma')(gray / 255.0)[..., :3]
        return (colored * 255).astype(np.uint8)
    except Exception:
        return np.stack([gray, gray, gray], axis=-1)


def select_first_batch(arr):
    if arr.ndim >= 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def make_map(arr, kind, camera_index):
    arr = select_first_batch(arr)
    if kind == 'bev':
        # Accept [C,H,W], [B,C,H,W], or [H,W].
        if arr.ndim == 3:
            return arr.mean(axis=0)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'Unsupported BEV shape: {arr.shape}')

    if kind in ['depth_argmax', 'depth_conf']:
        # Accept [Ncam,D,H,W], [D,H,W], or [B,Ncam,D,H,W].
        if arr.ndim == 5:
            arr = arr[0, camera_index]
        elif arr.ndim == 4:
            arr = arr[camera_index]
        if arr.ndim != 3:
            raise ValueError(f'Unsupported depth shape: {arr.shape}')
        # If logits were saved, softmax approximately before visualization.
        arr = arr - arr.max(axis=0, keepdims=True)
        prob = np.exp(arr)
        prob = prob / np.maximum(prob.sum(axis=0, keepdims=True), 1e-12)
        if kind == 'depth_argmax':
            return prob.argmax(axis=0)
        return prob.max(axis=0)

    if kind == 'heatmap':
        # Accept [task, cls, H, W], [cls,H,W], [H,W], or [B,cls,H,W].
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.max(axis=0)
        if arr.ndim == 3:
            return arr.max(axis=0)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'Unsupported heatmap shape: {arr.shape}')

    raise ValueError(kind)


def main():
    args = parse_args()
    arr = load_tensor(args.tensor)
    vis = make_map(arr, args.kind, args.camera_index)
    image = Image.fromarray(colorize(vis))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f'Wrote {args.kind} visualization to {out}, input shape={arr.shape}')


if __name__ == '__main__':
    main()
