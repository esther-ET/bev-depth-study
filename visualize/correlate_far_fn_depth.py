import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from nuscenes import NuScenes
from pyquaternion import Quaternion

from visualize.analyze_depth_vs_lidar import (CAMERAS, build_dataset,
                                               find_sample_index,
                                               summarize_errors)
from visualize.common import (box_in_camera, ego_distance, get_sample_gt_boxes,
                              load_prediction_json, match_predictions_to_gt,
                              prediction_to_box, project_box_to_image,
                              sample_camera_data)


DETECTION_CLASSES = {
    'car',
    'truck',
    'bus',
    'trailer',
    'construction_vehicle',
    'pedestrian',
    'motorcycle',
    'bicycle',
    'traffic_cone',
    'barrier',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Correlate far FN GT boxes with local depth prediction errors.')
    parser.add_argument('--exp',
                        default='bevdepth.exps.nuscenes.mv.'
                        'bev_depth_lss_r50_256x704_128x128_24e_2key')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--data-root', default='data/nuScenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--match-dist', type=float, default=2.0)
    parser.add_argument('--far-dist', type=float, default=40.0)
    parser.add_argument('--max-dist', type=float, default=None)
    parser.add_argument('--max-cases', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default='outputs/far_fn_depth_correlation.json')
    return parser.parse_args()


def val_ida_params(ida_aug_conf):
    raw_h, raw_w = ida_aug_conf['H'], ida_aug_conf['W']
    final_h, final_w = ida_aug_conf['final_dim']
    resize = max(final_h / raw_h, final_w / raw_w)
    new_w, new_h = int(raw_w * resize), int(raw_h * resize)
    crop_h = int((1 - np.mean(ida_aug_conf['bot_pct_lim'])) * new_h) - final_h
    crop_w = int(max(0, new_w - final_w) / 2)
    crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
    return resize, crop, (final_h, final_w)


def transform_points_to_model_image(points, resize, crop):
    out = points.copy()
    out[0, :] = out[0, :] * resize - crop[0]
    out[1, :] = out[1, :] * resize - crop[1]
    return out


def model_bbox_from_projected(points, resize, crop, final_dim, downsample):
    final_h, final_w = final_dim
    points = transform_points_to_model_image(points, resize, crop)
    xs, ys = points[0, :], points[1, :]
    x1, x2 = np.floor(xs.min()), np.ceil(xs.max())
    y1, y2 = np.floor(ys.min()), np.ceil(ys.max())
    if x2 < 0 or y2 < 0 or x1 >= final_w or y1 >= final_h:
        return None
    x1 = int(np.clip(x1, 0, final_w - 1))
    x2 = int(np.clip(x2, 0, final_w - 1))
    y1 = int(np.clip(y1, 0, final_h - 1))
    y2 = int(np.clip(y2, 0, final_h - 1))
    fx1 = max(0, x1 // downsample)
    fx2 = min(final_w // downsample - 1, x2 // downsample)
    fy1 = max(0, y1 // downsample)
    fy2 = min(final_h // downsample - 1, y2 // downsample)
    if fx2 < fx1 or fy2 < fy1:
        return None
    return dict(
        image_bbox=[x1, y1, x2, y2],
        feature_bbox=[fx1, fy1, fx2, fy2],
    )


def box_center_camera_depth(box_global, calibrated, ego_pose):
    box_cam = box_in_camera(box_global, calibrated, ego_pose)
    return float(box_cam.center[2])


def find_far_fn_cases(nusc, predictions, score_thr, match_dist, far_dist,
                      max_dist, max_cases):
    cases = []
    for sample_token, sample_preds in predictions.items():
        gt_boxes = [
            box for box in get_sample_gt_boxes(nusc, sample_token)
            if box.label in DETECTION_CLASSES
        ]
        pred_boxes = [
            prediction_to_box(pred)
            for pred in sample_preds
            if pred['detection_score'] >= score_thr
            and pred['detection_name'] in DETECTION_CLASSES
        ]
        _, fn_ids, _ = match_predictions_to_gt(
            gt_boxes,
            pred_boxes,
            distance_threshold=match_dist,
            class_aware=True,
        )
        for gi in fn_ids:
            gt = gt_boxes[gi]
            dist = ego_distance(nusc, sample_token, gt)
            if dist < far_dist:
                continue
            if max_dist is not None and dist > max_dist:
                continue
            cases.append(
                dict(
                    sample_token=sample_token,
                    gt_token=getattr(gt, 'token', ''),
                    category=gt.label,
                    ego_distance=dist,
                    visibility_token=getattr(gt, 'visibility_token', ''),
                ))
    cases.sort(key=lambda x: -x['ego_distance'])
    return cases[:max_cases]


def load_depth_maps(model, dataset, sample_token, device):
    sample_idx = find_sample_index(dataset, sample_token)
    from bevdepth.datasets.nusc_det_dataset import collate_fn

    batch = collate_fn([dataset[sample_idx]], is_return_depth=True)
    sweep_imgs, mats, _, _, _, _, depth_labels = batch
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
    pred_depth = bin_depths[pred_bin]
    conf = depth_prob.max(dim=1).values
    return gt_depth, pred_depth, conf, fg_mask


def analyze_case(nusc, model, depth_cache, case):
    sample_token = case['sample_token']
    gt_box = nusc.get_box(case['gt_token'])
    gt_box.label = case['category']

    gt_depth, pred_depth, conf, fg_mask = depth_cache[sample_token]
    resize, crop, final_dim = val_ida_params(model.ida_aug_conf)
    result = dict(case)
    result['camera_regions'] = []

    for cam_idx, camera in enumerate(CAMERAS):
        _, calibrated, ego_pose, image_path = sample_camera_data(
            nusc, sample_token, camera)
        from PIL import Image
        with Image.open(image_path) as image:
            projected = project_box_to_image(gt_box, calibrated, ego_pose,
                                             image.size)
        if projected is None:
            continue
        bbox = model_bbox_from_projected(projected, resize, crop, final_dim,
                                         model.downsample_factor)
        if bbox is None:
            continue
        fx1, fy1, fx2, fy2 = bbox['feature_bbox']
        region_mask = fg_mask[cam_idx, fy1:fy2 + 1, fx1:fx2 + 1]
        signed = (pred_depth[cam_idx, fy1:fy2 + 1, fx1:fx2 + 1] -
                  gt_depth[cam_idx, fy1:fy2 + 1, fx1:fx2 + 1])
        abs_err = signed.abs()
        region_conf = conf[cam_idx, fy1:fy2 + 1, fx1:fx2 + 1]
        valid_signed = signed[region_mask].detach().cpu().numpy()
        valid_abs = abs_err[region_mask].detach().cpu().numpy()
        valid_conf = region_conf[region_mask].detach().cpu().numpy()
        summary = summarize_errors(valid_abs, valid_signed)
        summary['mean_conf'] = (float(valid_conf.mean())
                                if valid_conf.size else None)
        summary['predicted_nearer_ratio'] = (
            float((valid_signed < -2.0).mean()) if valid_signed.size else None)
        summary['box_center_camera_depth'] = box_center_camera_depth(
            gt_box, calibrated, ego_pose)
        summary.update(bbox)
        summary['camera'] = camera
        result['camera_regions'].append(summary)

    valid_regions = [r for r in result['camera_regions'] if r.get('count', 0)]
    if valid_regions:
        total_count = sum(r['count'] for r in valid_regions)
        result['depth_region_count'] = total_count
        result['depth_region_mae'] = sum(r['mae'] * r['count']
                                         for r in valid_regions) / total_count
        result['depth_region_signed_error'] = sum(
            r.get('mean_signed_error', 0.0) * r['count']
            for r in valid_regions) / total_count
        result['depth_predicted_nearer_ratio'] = sum(
            (r.get('predicted_nearer_ratio') or 0.0) * r['count']
            for r in valid_regions) / total_count
    else:
        result['depth_region_count'] = 0
        result['depth_region_mae'] = None
        result['depth_region_signed_error'] = None
        result['depth_predicted_nearer_ratio'] = None
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

    nusc = NuScenes(version=args.version,
                    dataroot=args.data_root,
                    verbose=False)
    predictions = load_prediction_json(args.pred)
    dataset = build_dataset(model)

    cases = find_far_fn_cases(nusc, predictions, args.score_thr,
                              args.match_dist, args.far_dist, args.max_dist,
                              args.max_cases)
    depth_cache = {}
    for sample_token in sorted({case['sample_token'] for case in cases}):
        depth_cache[sample_token] = load_depth_maps(model, dataset,
                                                    sample_token, device)
    analyzed = [analyze_case(nusc, model, depth_cache, case) for case in cases]
    with_depth = [case for case in analyzed if case['depth_region_count'] > 0]
    predicted_nearer = [
        case for case in with_depth
        if case['depth_region_signed_error'] is not None
        and case['depth_region_signed_error'] < -2.0
    ]

    report = dict(
        exp=args.exp,
        ckpt=args.ckpt,
        pred=args.pred,
        score_thr=args.score_thr,
        match_dist=args.match_dist,
        far_dist=args.far_dist,
        max_dist=args.max_dist,
        num_cases=len(analyzed),
        num_cases_with_lidar_depth_region=len(with_depth),
        num_cases_predicted_nearer_by_2m=len(predicted_nearer),
        predicted_nearer_ratio=(len(predicted_nearer) / len(with_depth)
                                if with_depth else None),
        mean_region_signed_error=(float(np.mean([
            case['depth_region_signed_error'] for case in with_depth
        ])) if with_depth else None),
        mean_region_mae=(float(np.mean([
            case['depth_region_mae'] for case in with_depth
        ])) if with_depth else None),
        cases=analyzed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f'Wrote far FN depth correlation to {out_path}')


if __name__ == '__main__':
    main()
