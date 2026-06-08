import argparse
from pathlib import Path

from nuscenes import NuScenes

from visualize.common import (CAMERAS, COLORS, draw_boxes_on_camera,
                              get_sample_gt_boxes, load_prediction_json,
                              match_predictions_to_gt, prediction_to_box)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize TP/FP/FN 3D boxes on nuScenes camera images.')
    parser.add_argument('--data-root', default='data/nuScenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--sample-token', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--match-dist', type=float, default=2.0)
    parser.add_argument('--out-dir', default='outputs/vis_compare')
    return parser.parse_args()


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version,
                    dataroot=args.data_root,
                    verbose=False)
    predictions = load_prediction_json(args.pred)

    gt_boxes = get_sample_gt_boxes(nusc, args.sample_token)
    pred_boxes = [
        prediction_to_box(pred)
        for pred in predictions.get(args.sample_token, [])
        if pred['detection_score'] >= args.score_thr
    ]

    matches, fn_ids, fp_ids = match_predictions_to_gt(
        gt_boxes,
        pred_boxes,
        distance_threshold=args.match_dist,
        class_aware=True,
    )

    boxes_with_style = []
    for gi, pi, dist in matches:
        pred = pred_boxes[pi]
        boxes_with_style.append(
            (pred, COLORS['tp'], f'TP {pred.label} {pred.score:.2f}', 3))
    for gi in fn_ids:
        gt = gt_boxes[gi]
        boxes_with_style.append((gt, COLORS['fn'], f'FN {gt.label}', 4))
    for pi in fp_ids:
        pred = pred_boxes[pi]
        boxes_with_style.append(
            (pred, COLORS['fp'], f'FP {pred.label} {pred.score:.2f}', 3))

    sample_dir = Path(args.out_dir) / args.sample_token
    for camera in CAMERAS:
        draw_boxes_on_camera(
            nusc,
            args.sample_token,
            camera,
            boxes_with_style,
            sample_dir / f'{camera}.jpg',
        )

    print(
        f'Wrote comparison to {sample_dir} '
        f'(TP={len(matches)}, FP={len(fp_ids)}, FN={len(fn_ids)})')


if __name__ == '__main__':
    main()
