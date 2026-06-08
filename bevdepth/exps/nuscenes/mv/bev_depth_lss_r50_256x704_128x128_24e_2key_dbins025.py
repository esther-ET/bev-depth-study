# Copyright (c) Megvii Inc. All rights reserved.
from copy import deepcopy

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.exps.nuscenes.base_exp import backbone_conf as base_backbone_conf
from bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        self.flexible_frustum_load = kwargs.get('evaluate',
                                                False) or kwargs.get(
                                                    'predict', False)
        backbone_conf = deepcopy(kwargs.pop('backbone_conf',
                                            base_backbone_conf))
        backbone_conf['d_bound'] = [2.0, 58.0, 0.25]
        super().__init__(backbone_conf=backbone_conf, **kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)

    def load_state_dict(self, state_dict, strict=True):
        if self.flexible_frustum_load:
            state_dict = dict(state_dict)
            state_dict.pop('model.backbone.frustum', None)
            strict = False
        return super().load_state_dict(state_dict, strict=strict)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_2key_dbins025')
