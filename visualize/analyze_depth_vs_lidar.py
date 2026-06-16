import argparse
import importlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from visualize.dump_bev_depth_outputs import denormalize_image
from visualize.visualize_feature_depth import colorize


CAMERAS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare BEVDepth depth prediction with LiDAR-projected depth labels.')
    parser.add_argument('--exp',
                        default='bevdepth.exps.nuscenes.mv.'
                        'bev_depth_lss_r50_256x704_128x128_24e_2key')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-root', default='data/nuScenes')
    parser.add_argument('--sample-token', default=None)
    parser.add_argument('--max-samples', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', default='outputs/depth_analysis')
    return parser.parse_args()


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
        return_depth=True,
        use_fusion=model.use_fusion,
    )


def find_sample_index(dataset, sample_token):
    for idx, info in enumerate(dataset.infos):
        if info['sample_token'] == sample_token:
            return idx
    raise ValueError(f'sample_token not found in dataset: {sample_token}')


def save_color_map(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize(arr)).save(path)


def save_masked_error(err, mask, path, vmax=None):
    out = np.zeros((*err.shape, 3), dtype=np.uint8)
    if mask.any():
        vals = err[mask]
        if vmax is None:
            vmax = max(float(np.percentile(vals, 95)), 1e-6)
        norm = np.clip(err / vmax, 0, 1)
        out[...] = colorize(norm)
        out[~mask] = 0
    Image.fromarray(out).save(path)


def summarize_errors(errors, signed_errors=None):
    errors = np.asarray(errors, dtype=np.float32)
    if errors.size == 0:
        return dict(count=0)
    summary = dict(
        count=int(errors.size),
        mae=float(errors.mean()),
        median=float(np.median(errors)),
        p75=float(np.percentile(errors, 75)),
        p90=float(np.percentile(errors, 90)),
        within_1m=float((errors <= 1.0).mean()),
        within_2m=float((errors <= 2.0).mean()),
        within_4m=float((errors <= 4.0).mean()),
    )
    if signed_errors is not None:
        signed_errors = np.asarray(signed_errors, dtype=np.float32)
        summary['mean_signed_error'] = float(signed_errors.mean())
    return summary


def summarize_by_depth_ranges(gt_depth, pred_depth, mask):
    ranges = [(2, 10), (10, 20), (20, 40), (40, 58)]
    out = {}
    signed = pred_depth - gt_depth
    abs_err = signed.abs()
    for start, end in ranges:
        cur_mask = mask & (gt_depth >= start) & (gt_depth < end)
        key = f'{start}-{end}m'
        out[key] = summarize_errors(
            abs_err[cur_mask].detach().cpu().numpy(),
            signed[cur_mask].detach().cpu().numpy(),
        )
    return out


def analyze_one(model, dataset, sample_idx, device, out_dir, save_images):
    batch = collate_fn([dataset[sample_idx]], is_return_depth=True)
    sweep_imgs, mats, _, img_metas, _, _, depth_labels = batch
    sweep_imgs = sweep_imgs.to(device)
    mats = {key: value.to(device) for key, value in mats.items()}
    depth_labels = depth_labels.to(device)
    if depth_labels.ndim == 5:
        depth_labels = depth_labels[:, 0, ...]

    with torch.no_grad():
        _, depth_prob = model.model.backbone._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats,
            is_return_depth=True,
        )

    gt_onehot = model.get_downsampled_gt_depth(depth_labels).view(
        depth_prob.shape[0], depth_prob.shape[2], depth_prob.shape[3],
        depth_prob.shape[1]).permute(0, 3, 1, 2).contiguous()
    fg_mask = gt_onehot.max(dim=1).values > 0

    gt_bin = gt_onehot.argmax(dim=1)
    pred_bin = depth_prob.argmax(dim=1)
    bin_values = torch.arange(model.depth_channels,
                              device=device,
                              dtype=depth_prob.dtype)
    bin_depths = model.dbound[0] + bin_values * model.dbound[2]
    gt_depth = bin_depths[gt_bin]
    pred_depth_argmax = bin_depths[pred_bin]
    pred_depth_expect = (depth_prob * bin_depths.view(1, -1, 1, 1)).sum(dim=1)
    conf = depth_prob.max(dim=1).values

    argmax_err = (pred_depth_argmax - gt_depth).abs()
    expect_err = (pred_depth_expect - gt_depth).abs()
    bin_err = (pred_bin - gt_bin).abs()

    sample_token = img_metas[0]['token']
    result = dict(sample_token=sample_token, cameras={})
    all_argmax = []
    all_expect = []
    all_bin = []
    all_conf = []

    sample_dir = Path(out_dir) / sample_token
    if save_images:
        sample_dir.mkdir(parents=True, exist_ok=True)

    for cam_idx, cam_name in enumerate(CAMERAS):
        mask = fg_mask[cam_idx].detach().cpu().numpy().astype(bool)
        cam_argmax = argmax_err[cam_idx][fg_mask[cam_idx]].detach().cpu().numpy()
        cam_expect = expect_err[cam_idx][fg_mask[cam_idx]].detach().cpu().numpy()
        cam_bin = bin_err[cam_idx][fg_mask[cam_idx]].detach().cpu().numpy()
        cam_conf = conf[cam_idx][fg_mask[cam_idx]].detach().cpu().numpy()

        all_argmax.extend(cam_argmax.tolist())
        all_expect.extend(cam_expect.tolist())
        all_bin.extend(cam_bin.tolist())
        all_conf.extend(cam_conf.tolist())

        cam_summary = summarize_errors(
            cam_argmax,
            (pred_depth_argmax[cam_idx][fg_mask[cam_idx]] -
             gt_depth[cam_idx][fg_mask[cam_idx]]).detach().cpu().numpy(),
        )
        cam_summary['expected_mae'] = summarize_errors(cam_expect).get('mae', None)
        cam_summary['bin_mae'] = float(cam_bin.mean()) if cam_bin.size else None
        cam_summary['mean_conf'] = float(cam_conf.mean()) if cam_conf.size else None
        cam_summary['by_gt_depth'] = summarize_by_depth_ranges(
            gt_depth[cam_idx],
            pred_depth_argmax[cam_idx],
            fg_mask[cam_idx],
        )
        result['cameras'][cam_name] = cam_summary

        if save_images:
            Image.fromarray(denormalize_image(sweep_imgs[0, 0, cam_idx])).save(
                sample_dir / f'{cam_idx}_{cam_name}_input.jpg')
            save_color_map(gt_depth[cam_idx].detach().cpu().numpy(),
                           sample_dir / f'{cam_idx}_{cam_name}_gt_depth.jpg')
            save_color_map(pred_depth_argmax[cam_idx].detach().cpu().numpy(),
                           sample_dir / f'{cam_idx}_{cam_name}_pred_argmax_depth.jpg')
            save_color_map(pred_depth_expect[cam_idx].detach().cpu().numpy(),
                           sample_dir / f'{cam_idx}_{cam_name}_pred_expected_depth.jpg')
            save_color_map(conf[cam_idx].detach().cpu().numpy(),
                           sample_dir / f'{cam_idx}_{cam_name}_depth_conf.jpg')
            save_masked_error(argmax_err[cam_idx].detach().cpu().numpy(), mask,
                              sample_dir / f'{cam_idx}_{cam_name}_argmax_error.jpg')
            save_masked_error(expect_err[cam_idx].detach().cpu().numpy(), mask,
                              sample_dir / f'{cam_idx}_{cam_name}_expected_error.jpg')

    result['overall'] = summarize_errors(all_argmax)
    result['overall']['expected'] = summarize_errors(all_expect)
    result['overall']['bin_mae'] = float(np.mean(all_bin)) if all_bin else None
    result['overall']['mean_conf'] = float(np.mean(all_conf)) if all_conf else None
    return result


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
    if args.sample_token:
        indices = [find_sample_index(dataset, args.sample_token)]
    else:
        indices = list(range(min(args.max_samples, len(dataset))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sample_idx in enumerate(indices):
        results.append(
            analyze_one(model,
                        dataset,
                        sample_idx,
                        device,
                        out_dir,
                        save_images=(i == 0)))

    all_argmax = []
    all_expect = []
    for item in results:
        # Reconstruct aggregate from per-camera summaries approximately is not
        # exact, so keep report as per-sample plus count-weighted top-level not needed.
        all_argmax.append(item['overall']['mae'])
        all_expect.append(item['overall']['expected']['mae'])

    report = dict(
        exp=args.exp,
        ckpt=args.ckpt,
        dbound=model.dbound,
        depth_channels=model.depth_channels,
        num_samples=len(results),
        sample_results=results,
        mean_sample_argmax_mae=float(np.mean(all_argmax)) if all_argmax else None,
        mean_sample_expected_mae=float(np.mean(all_expect)) if all_expect else None,
    )
    out_path = out_dir / 'depth_vs_lidar_summary.json'
    with out_path.open('w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f'Wrote depth analysis to {out_path}')


if __name__ == '__main__':
    main()
