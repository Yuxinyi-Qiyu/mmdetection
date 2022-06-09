# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from mmdet.models.dense_heads import YOLOXHead
from ..utils.usconv import set_channels, make_divisible

# import sys
# sys.path.append("../backbone/")
from ..utils import USConv2d, USBatchNorm2d


@HEADS.register_module()
class YOLOXHead_Searchable(YOLOXHead):
    def set_arch(self, arch, divisor=8):
        widen_factor_head = arch['widen_factor_head']
        if isinstance(widen_factor_head, (int, float)):
            widen_factor_head = [widen_factor_head] * 2
        in_channels = make_divisible(self.in_channels * widen_factor_head[0], divisor)
        feat_channels = make_divisible(self.in_channels * widen_factor_head[1], divisor)

        set_channels(self.multi_level_cls_convs, feat_channels, feat_channels)
        set_channels(self.multi_level_reg_convs, feat_channels, feat_channels)

        set_channels(self.multi_level_conv_cls, in_channels=feat_channels)
        set_channels(self.multi_level_conv_reg, in_channels=feat_channels)
        set_channels(self.multi_level_conv_obj, in_channels=feat_channels)

        for i, _ in enumerate(self.strides):
            self.multi_level_cls_convs[i][0].conv.in_channels = in_channels
            self.multi_level_reg_convs[i][0].conv.in_channels = in_channels


