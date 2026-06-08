import argparse
import json
from pathlib import Path

from nuscenes import NuScenes

from visualize.common import (box_volume, ego_distance, get_sample_gt_boxes,
                              load_prediction_json, match_predictions_to_gt,
                              prediction_to_box)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Mine simple nuScenes detection bad cases.')
    parser.add_argument('--data-root', default='data/nuScenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--pred', required=True)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--match-dist', type=float, default=2.0)
    parser.add_argument('--far-dist', type=float, default=40.0)
    parser.add_argument('--small-volume', type=float, default=2.0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', default='outputs/bad_cases.json')
    return parser.parse_args()


def add_case(cases, **kwargs):
    cases.append(kwargs)


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version,
                    dataroot=args.data_root,
                    verbose=False)
    predictions = load_prediction_json(args.pred)

    cases = []
    sample_tokens = list(predictions.keys())
    if args.limit is not None:
        sample_tokens = sample_tokens[:args.limit]

    for sample_token in sample_tokens:
        gt_boxes = get_sample_gt_boxes(nusc, sample_token)
        pred_boxes = [
            prediction_to_box(pred)
            for pred in predictions.get(sample_token, [])
            if pred['detection_score'] >= args.score_thr
        ]
        matches, fn_ids, fp_ids = match_predictions_to_gt(
            gt_boxes,
            pred_boxes,
            distance_threshold=args.match_dist,
            class_aware=True,
        )
        for gi, pi, dist in matches:
            gt = gt_boxes[gi]
            pred = pred_boxes[pi]
            distance = ego_distance(nusc, sample_token, gt)
            volume = box_volume(gt)
            tags = []
            if distance >= args.far_dist:
                tags.append('far_tp')
            if volume <= args.small_volume:
                tags.append('small_tp')
            if tags:
                add_case(
                    cases,
                    sample_token=sample_token,
                    error_type='tp_hard_case',
                    tags=tags,
                    category=gt.label,
                    distance=distance,
                    volume=volume,
                    center_error=dist,
                    pred_score=float(pred.score),
                    gt_token=getattr(gt, 'token', ''),
                    visibility_token=getattr(gt, 'visibility_token', ''),
                )

        for gi in fn_ids:
            gt = gt_boxes[gi]
            distance = ego_distance(nusc, sample_token, gt)
            volume = box_volume(gt)
            tags = ['missed_gt']
            if distance >= args.far_dist:
                tags.append('far')
            if volume <= args.small_volume:
                tags.append('small')
            visibility = getattr(gt, 'visibility_token', '')
            if visibility in ['1', '2']:
                tags.append('low_visibility')
            add_case(
                cases,
                sample_token=sample_token,
                error_type='fn',
                tags=tags,
                category=gt.label,
                distance=distance,
                volume=volume,
                gt_token=getattr(gt, 'token', ''),
                visibility_token=visibility,
            )

        for pi in fp_ids:
            pred = pred_boxes[pi]
            distance = ego_distance(nusc, sample_token, pred)
            volume = box_volume(pred)
            tags = ['false_positive']
            if distance >= args.far_dist:
                tags.append('far')
            if volume <= args.small_volume:
                tags.append('small')
            add_case(
                cases,
                sample_token=sample_token,
                error_type='fp',
                tags=tags,
                category=pred.label,
                distance=distance,
                volume=volume,
                pred_score=float(pred.score),
            )

    cases.sort(key=lambda x: (
        x['error_type'] != 'fn',
        -x.get('distance', 0.0),
        x.get('pred_score', 0.0),
    ))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(cases, f, indent=2)
    print(f'Wrote {len(cases)} bad cases to {out_path}')


if __name__ == '__main__':
    main()
