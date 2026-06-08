# Copyright (c) Megvii Inc. All rights reserved.
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        self.flexible_frustum_load = kwargs.get('evaluate',
                                                False) or kwargs.get(
                                                    'predict', False)
        super().__init__(**kwargs)
        final_dim = (192, 640)
        self.backbone_conf['final_dim'] = final_dim
        self.ida_aug_conf['final_dim'] = final_dim
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
            'bev_depth_lss_r50_192x640_128x128_24e_2key')
