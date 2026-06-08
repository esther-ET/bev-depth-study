import argparse
import importlib
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from visualize.visualize_feature_depth import colorize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one BEVDepth sample and dump BEV/depth/heatmap visualizations.')
    parser.add_argument('--exp',
                        default='bevdepth.exps.nuscenes.mv.'
                        'bev_depth_lss_r50_256x704_128x128_24e_2key')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--sample-token', required=True)
    parser.add_argument('--data-root', default='data/nuScenes')
    parser.add_argument('--out-dir', default='outputs/vis_internal')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-tensors', action='store_true')
    return parser.parse_args()


def denormalize_image(tensor):
    arr = tensor.detach().cpu().float().numpy()
    arr = arr.transpose(1, 2, 0)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    arr = arr * std + mean
    return arr.clip(0, 255).astype(np.uint8)


def save_color_map(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize(arr)).save(path)


def build_dataset(model):
    return NuscDetDataset(
        ida_aug_conf=model.ida_aug_conf,
        bda_aug_conf=model.bda_aug_conf,
        classes=model.class_names,
        data_root=model.data_root,
        info_paths=model.val_info_paths,
        is_train=False,
        img_conf=model.img_conf,
        num_sweeps=model.num_sweeps,
        sweep_idxes=model.sweep_idxes,
        key_idxes=model.key_idxes,
        return_depth=False,
        use_fusion=model.use_fusion,
    )


def find_sample_index(dataset, sample_token):
    for idx, info in enumerate(dataset.infos):
        if info['sample_token'] == sample_token:
            return idx
    raise ValueError(f'sample_token not found in dataset: {sample_token}')


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    module = importlib.import_module(args.exp)
    model = module.BEVDepthLightningModel(gpus=1,
                                          batch_size_per_device=1,
                                          data_root=args.data_root)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    dataset = build_dataset(model)
    sample_idx = find_sample_index(dataset, args.sample_token)
    batch = collate_fn([dataset[sample_idx]], is_return_depth=False)
    sweep_imgs, mats, timestamps, img_metas, _, _ = batch
    sweep_imgs = sweep_imgs.to(device)
    timestamps = timestamps.to(device)
    mats = {key: value.to(device) for key, value in mats.items()}

    out_dir = Path(args.out_dir) / args.sample_token
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        backbone = model.model.backbone
        key_splat, depth_prob = backbone._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats,
            is_return_depth=True,
        )
        splat_features = [key_splat]
        for sweep_idx in range(1, sweep_imgs.shape[1]):
            splat = backbone._forward_single_sweep(
                sweep_idx,
                sweep_imgs[:, sweep_idx:sweep_idx + 1, ...],
                mats,
                is_return_depth=False,
            )
            splat_features.append(splat)
        bev_feature = torch.cat(splat_features, dim=1)
        preds = model.model.head(bev_feature)

    for sweep_idx, splat in enumerate(splat_features):
        splat_map = splat[0].abs().mean(dim=0).detach().cpu().numpy()
        save_color_map(splat_map, out_dir / f'splat_sweep{sweep_idx}_mean.jpg')
        splat_max = splat[0].abs().max(dim=0).values.detach().cpu().numpy()
        save_color_map(splat_max, out_dir / f'splat_sweep{sweep_idx}_max.jpg')

    bev_map = bev_feature[0].abs().mean(dim=0).detach().cpu().numpy()
    save_color_map(bev_map, out_dir / 'bev_feature_mean.jpg')

    heatmaps = []
    for task_id, task_pred in enumerate(preds):
        heatmap = task_pred[0]['heatmap'].sigmoid()[0]
        task_map = heatmap.max(dim=0).values.detach().cpu().numpy()
        heatmaps.append(task_map)
        save_color_map(task_map, out_dir / f'head_heatmap_task{task_id}.jpg')
    save_color_map(np.max(np.stack(heatmaps), axis=0),
                   out_dir / 'head_heatmap_all_tasks.jpg')

    depth_prob = depth_prob.detach().cpu()
    num_cams = depth_prob.shape[0]
    for cam_idx in range(num_cams):
        prob = depth_prob[cam_idx]
        depth_argmax = prob.argmax(dim=0).numpy()
        depth_conf = prob.max(dim=0).values.numpy()
        save_color_map(depth_argmax, out_dir / f'depth_argmax_cam{cam_idx}.jpg')
        save_color_map(depth_conf, out_dir / f'depth_conf_cam{cam_idx}.jpg')

    # Save the normalized input views for quick alignment checks.
    for cam_idx in range(sweep_imgs.shape[2]):
        image = denormalize_image(sweep_imgs[0, 0, cam_idx])
        Image.fromarray(image).save(out_dir / f'input_cam{cam_idx}.jpg')

    if args.save_tensors:
        torch.save({
            'bev_feature': bev_feature.detach().cpu(),
            'depth_prob': depth_prob,
            'preds': preds,
            'img_metas': img_metas,
        }, out_dir / 'internal_outputs.pt')

    print(f'Wrote BEV/depth/head visualizations to {out_dir}')
    print(f'bev_feature={tuple(bev_feature.shape)}, '
          f'depth_prob={tuple(depth_prob.shape)}, tasks={len(preds)}, '
          f'splats={[tuple(x.shape) for x in splat_features]}')


if __name__ == '__main__':
    main()
