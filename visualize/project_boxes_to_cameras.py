import argparse
from pathlib import Path

from nuscenes import NuScenes

from visualize.common import (CAMERAS, COLORS, draw_boxes_on_camera,
                              get_sample_gt_boxes, load_prediction_json,
                              prediction_to_box)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Project nuScenes 3D boxes to the six camera images.')
    parser.add_argument('--data-root', default='data/nuScenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--sample-token', required=True)
    parser.add_argument('--pred', default=None)
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--box-source',
                        choices=['gt', 'pred', 'both'],
                        default='both')
    parser.add_argument('--out-dir', default='outputs/vis_boxes')
    return parser.parse_args()


def main():
    args = parse_args()
    nusc = NuScenes(version=args.version,
                    dataroot=args.data_root,
                    verbose=False)

    boxes_with_style = []
    if args.box_source in ['gt', 'both']:
        for box in get_sample_gt_boxes(nusc, args.sample_token):
            label = f'GT {box.label}'
            boxes_with_style.append((box, COLORS['gt'], label, 3))

    if args.box_source in ['pred', 'both']:
        if args.pred is None:
            raise ValueError('--pred is required when using pred boxes.')
        predictions = load_prediction_json(args.pred)
        for pred in predictions.get(args.sample_token, []):
            if pred['detection_score'] < args.score_thr:
                continue
            box = prediction_to_box(pred)
            label = f'{box.label} {box.score:.2f}'
            boxes_with_style.append((box, COLORS['pred'], label, 2))

    sample_dir = Path(args.out_dir) / args.sample_token
    for camera in CAMERAS:
        draw_boxes_on_camera(
            nusc,
            args.sample_token,
            camera,
            boxes_with_style,
            sample_dir / f'{camera}.jpg',
        )
    print(f'Wrote camera projections to {sample_dir}')


if __name__ == '__main__':
    main()
