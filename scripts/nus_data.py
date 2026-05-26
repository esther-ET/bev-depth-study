from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.0-trainval',
    dataroot='./data/nuScenes/',
    verbose=True,
)
# 以上实例化会出触发Loading NuScenes tables for version v1.0-trainval...的过程

# 取得一张图片
sample = nusc.sample[0]
print(sample.keys())
# dict_keys(['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns'])
print(sample)
# {'token': 'e93e98b63d3b40209056d129dc53ceee', 'timestamp': 1531883530449377, 'prev': '', 'next': '14d5adfe50bb4445bc3aa5fe607691a8', 'scene_token': '73030fb67d3c46cfb5e590168088ae39', 'data': {'RADAR_FRONT': 'bddd80ae33ec4e32b27fdb3c1160a30e', 'RADAR_FRONT_LEFT': '1a08aec0958e42ebb37d26612a2cfc57', 'RADAR_FRONT_RIGHT': '282fa8d7a3f34b68b56fb1e22e697668', 'RADAR_BACK_LEFT': '05fc4678025246f3adf8e9b8a0a0b13b', 'RADAR_BACK_RIGHT': '31b8099fb1c44c6381c3c71b335750bb', 'LIDAR_TOP': '3388933b59444c5db71fade0bbfef470', 'CAM_FRONT': '020d7b4f858147558106c504f7f31bef', 'CAM_FRONT_RIGHT': '16d39ff22a8545b0a4ee3236a0fe1c20', 'CAM_BACK_RIGHT': 'ec7096278e484c9ebe6894a2ad5682e9', 'CAM_BACK': 'aab35aeccbda42de82b2ff5c278a0d48', 'CAM_BACK_LEFT': '86e6806d626b4711a6d0f5015b090116', 'CAM_FRONT_LEFT': '24332e9c554a406f880430f17771b608'}, 'anns': ['173a50411564442ab195e132472fde71', '5123ed5e450948ac8dc381772f2ae29a', 'acce0b7220754600b700257a1de1573d', '8d7cb5e96cae48c39ef4f9f75182013a', 'f64bfd3d4ddf46d7a366624605cb7e91', 'f9dba7f32ed34ee8adc92096af767868', '086e3f37a44e459987cde7a3ca273b5b', '3964235c58a745df8589b6a626c29985', '31a96b9503204a8688da75abcd4b56b2', 'b0284e14d17a444a8d0071bd1f03a0a2']}

cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
print(cam)
# {'token': '020d7b4f858147558106c504f7f31bef', 'sample_token': 'e93e98b63d3b40209056d129dc53ceee', 'ego_pose_token': '020d7b4f858147558106c504f7f31bef', 'calibrated_sensor_token': '2e64b091b3b146a390c2606b9081343c', 'timestamp': 1531883530412470, 'fileformat': 'jpg', 'is_key_frame': True, 'height': 900, 'width': 1600, 'filename': 'samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg', 'prev': '', 'next': 'caa2bfad0b8a4a8090cb0b803352cbc8', 'sensor_modality': 'camera', 'channel': 'CAM_FRONT'}

ego_pose = nusc.get('ego_pose', cam['ego_pose_token'])
print(ego_pose)
# {'token': '020d7b4f858147558106c504f7f31bef', 'timestamp': 1531883530412470, 'rotation': [-0.7530285141171715, -0.007718682910458633, 0.00863090844122062, -0.6578859979358822], 'translation': [1010.1102882349232, 610.6567106479714, 0.0]}

calib = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
print(calib)
# {'token': '2e64b091b3b146a390c2606b9081343c', 'sensor_token': '725903f5b62f56118f4094b46a4470d8', 'translation': [1.70079118954, 0.0159456324149, 1.51095763913], 'rotation': [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755], 'camera_intrinsic': [[1266.417203046554, 0.0, 816.2670197447984], [0.0, 1266.417203046554, 491.50706579294757], [0.0, 0.0, 1.0]]}





