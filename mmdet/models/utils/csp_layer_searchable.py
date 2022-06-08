# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.utils import CSPLayer

class CSPLayer_Searchable(CSPLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(in_channels, out_channels, expand_ratio, num_blocks, add_identity, use_depthwise, conv_cfg, norm_cfg, act_cfg, init_cfg)
        self.num_blocks = num_blocks

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        for block in self.blocks[:self.num_blocks]:
            x_main = block(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)
