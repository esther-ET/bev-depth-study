import json
import math
from pathlib import Path

import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion


CAMERAS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

FRONT_EDGES = [(0, 1), (1, 5), (5, 4), (4, 0)]

COLORS = {
    'gt': (40, 220, 90), # 绿色
    'tp': (60, 140, 255), # 蓝色
    'fp': (255, 70, 70), # 红色
    'fn': (255, 210, 50), # 黄色
    'pred': (60, 140, 255), # 蓝色
}


def load_prediction_json(pred_path):
    pred_path = Path(pred_path)
    with pred_path.open('r') as f:
        data = json.load(f)
    if 'results' not in data:
        raise ValueError(f'{pred_path} is not a nuScenes results json.')
    return data['results']


def prediction_to_box(pred):
    box = Box(
        center=pred['translation'],
        size=pred['size'],
        orientation=Quaternion(pred['rotation']),
        score=pred.get('detection_score', float('nan')),
        velocity=tuple(pred.get('velocity', [0.0, 0.0])) + (0.0,),
    )
    box.label = pred.get('detection_name', '')
    return box


def get_sample_gt_boxes(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = nusc.get_box(ann_token)
        box.label = category_to_detection_name(ann['category_name'])
        box.token = ann_token
        box.visibility_token = ann.get('visibility_token', '')
        boxes.append(box)
    return boxes


def category_to_detection_name(category_name):
    parts = category_name.split('.')
    if category_name.startswith('vehicle.car'):
        return 'car'
    if category_name.startswith('vehicle.truck'):
        return 'truck'
    if category_name.startswith('vehicle.bus'):
        return 'bus'
    if category_name.startswith('vehicle.trailer'):
        return 'trailer'
    if category_name.startswith('vehicle.construction'):
        return 'construction_vehicle'
    if category_name.startswith('human.pedestrian'):
        return 'pedestrian'
    if category_name.startswith('vehicle.motorcycle'):
        return 'motorcycle'
    if category_name.startswith('vehicle.bicycle'):
        return 'bicycle'
    if category_name.startswith('movable_object.trafficcone'):
        return 'traffic_cone'
    if category_name.startswith('movable_object.barrier'):
        return 'barrier'
    return parts[-1]


def sample_camera_data(nusc, sample_token, camera):
    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data'][camera])
    calibrated = nusc.get('calibrated_sensor',
                          sample_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    image_path = Path(nusc.dataroot) / sample_data['filename']
    return sample_data, calibrated, ego_pose, image_path


def box_in_camera(box_global, calibrated, ego_pose):
    box = box_global.copy()
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)
    box.translate(-np.array(calibrated['translation']))
    box.rotate(Quaternion(calibrated['rotation']).inverse)
    return box


def project_box_to_image(box_global,
                         calibrated,
                         ego_pose,
                         image_size,
                         min_z=1e-3):
    box_cam = box_in_camera(box_global, calibrated, ego_pose)
    corners = box_cam.corners()
    if not np.any(corners[2, :] > min_z):
        return None
    intrinsic = np.array(calibrated['camera_intrinsic'])
    points = view_points(corners, intrinsic, normalize=True)[:2, :]
    visible = corners[2, :] > min_z
    width, height = image_size
    in_frame = (
        (points[0, :] >= 0) & (points[0, :] < width) &
        (points[1, :] >= 0) & (points[1, :] < height) &
        visible
    )
    if not np.any(in_frame):
        return None
    return points


def draw_projected_box(draw, corners_2d, color, width=3, label=None):
    pts = [(float(corners_2d[0, i]), float(corners_2d[1, i]))
           for i in range(corners_2d.shape[1])]
    for edge in BOX_EDGES:
        draw.line([pts[edge[0]], pts[edge[1]]], fill=color, width=width)
    for edge in FRONT_EDGES:
        draw.line([pts[edge[0]], pts[edge[1]]], fill=color, width=width + 1)
    if label:
        x = min(p[0] for p in pts)
        y = min(p[1] for p in pts)
        draw.text((x, max(0, y - 12)), label, fill=color)


def draw_boxes_on_camera(nusc,
                         sample_token,
                         camera,
                         boxes_with_style,
                         out_path):
    _, calibrated, ego_pose, image_path = sample_camera_data(
        nusc, sample_token, camera)
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), camera, fill=(255, 255, 255))
    for box, color, label, width in boxes_with_style:
        corners = project_box_to_image(box, calibrated, ego_pose, image.size)
        if corners is None:
            continue
        draw_projected_box(draw, corners, color, width=width, label=label)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def ego_distance(nusc, sample_token, box_global):
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    delta = np.array(box_global.center[:2]) - np.array(ego_pose['translation'][:2])
    return float(np.linalg.norm(delta))


def center_distance(box_a, box_b):
    return float(np.linalg.norm(np.array(box_a.center[:2]) -
                                np.array(box_b.center[:2])))


def box_volume(box):
    return float(np.prod(box.wlh))

# 如何判断 TP FP FN
def match_predictions_to_gt(gt_boxes,
                            pred_boxes,
                            distance_threshold=2.0,
                            class_aware=True):
    matches = []
    used_gt = set()
    used_pred = set()
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if class_aware and gt.label != pred.label:
                continue
            candidates.append((center_distance(gt, pred), gi, pi))
    candidates.sort(key=lambda x: x[0])
    for dist, gi, pi in candidates:
        if dist > distance_threshold:
            continue
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, dist))
    fn = [i for i in range(len(gt_boxes)) if i not in used_gt]
    fp = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return matches, fn, fp

# token校验
def ensure_sample_tokens(nusc, sample_tokens=None, limit=None):
    if sample_tokens:
        tokens = list(sample_tokens)
    else:
        tokens = [sample['token'] for sample in nusc.sample]
    if limit is not None:
        tokens = tokens[:limit]
    return tokens
