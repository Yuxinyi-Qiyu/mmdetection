# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import CSPLayer


@NECKS.register_module()
class YOLOXPAFPN_tfs(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 widen_factor=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                 widen_factor_out=0.5,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN_tfs, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.widen_factor = widen_factor
        self.widen_factor_out = widen_factor_out

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        self.base_out_channels = 256
        new_out_channel = int(self.base_out_channels * self.widen_factor_out)
        self.new_out_channel = new_out_channel

        self.base_channels_backbone = [256, 512, 1024]
        self.base_channels_dict = {  # 之前写错了
            'reduce_layers0': 512,
            'reduce_layers1': 256,
            'top_down_blocks0': 512,
            'top_down_blocks1': 256,
            'downsamples0': 256,
            'downsamples1': 512,
            'bottom_up_blocks0': 512,
            'bottom_up_blocks1': 1024
        }
        # create factor_dictionary
        self.widen_factor_dict = {
            'reduce_layers0': widen_factor[0],
            'reduce_layers1': widen_factor[1],
            'top_down_blocks0': widen_factor[2],
            'top_down_blocks1': widen_factor[3],
            'downsamples0': widen_factor[4],
            'downsamples1': widen_factor[5],
            'bottom_up_blocks0': widen_factor[6],
            'bottom_up_blocks1': widen_factor[7],
        }

        channels_out_dict = {
            'reduce_layers0':int(self.widen_factor_dict['reduce_layers0'] * self.base_channels_dict['reduce_layers0']),
            'reduce_layers1': int(self.widen_factor_dict['reduce_layers1'] * self.base_channels_dict['reduce_layers1']),
            'top_down_blocks0': int(self.widen_factor_dict['top_down_blocks0'] * self.base_channels_dict['top_down_blocks0']),
            'top_down_blocks1': int(self.widen_factor_dict['top_down_blocks1'] * self.base_channels_dict['top_down_blocks1']),
            'downsamples0': int(self.widen_factor_dict['downsamples0'] * self.base_channels_dict['downsamples0']),
            'downsamples1': int(self.widen_factor_dict['downsamples1'] * self.base_channels_dict['downsamples1']),
            'bottom_up_blocks0': int(self.widen_factor_dict['bottom_up_blocks0'] * self.base_channels_dict['bottom_up_blocks0']),
            'bottom_up_blocks1': int(self.widen_factor_dict['bottom_up_blocks1'] * self.base_channels_dict['bottom_up_blocks1']),
        }

        channels_dict = {
            'reduce_layers0': [in_channels[2], channels_out_dict['reduce_layers0']],
            'reduce_layers1': [channels_out_dict['top_down_blocks0'], channels_out_dict['reduce_layers1']],
            'top_down_blocks0': [(in_channels[1] + channels_out_dict['reduce_layers0']), channels_out_dict['top_down_blocks0']],
            'top_down_blocks1': [(in_channels[0] + channels_out_dict['reduce_layers1']), channels_out_dict['top_down_blocks1']],
            'downsamples0': [channels_out_dict['top_down_blocks1'], channels_out_dict['downsamples0']],
            'downsamples1': [channels_out_dict['bottom_up_blocks0'], channels_out_dict['downsamples1']],
            'bottom_up_blocks0': [(channels_out_dict['reduce_layers1'] + channels_out_dict['downsamples0']), channels_out_dict['bottom_up_blocks0']],
            'bottom_up_blocks1': [(channels_out_dict['reduce_layers0'] + channels_out_dict['downsamples1']), channels_out_dict['bottom_up_blocks1']],
        }

        for idx in range(len(in_channels) - 1, 0, -1):
            layer_name_reduce = 'reduce_layers' + str(2 - idx)
            layer_name_td = 'top_down_blocks' + str(2 - idx)
            self.reduce_layers.append(
                ConvModule(
                    # in_channels[idx],
                    # in_channels[idx - 1],
                    channels_dict[layer_name_reduce][0],
                    channels_dict[layer_name_reduce][1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    # in_channels[idx - 1] * 2,
                    # in_channels[idx - 1],
                    channels_dict[layer_name_td][0],
                    channels_dict[layer_name_td][1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            layer_name_downsample = 'downsamples' + str(idx)
            layer_name_bu = 'bottom_up_blocks' + str(idx)
            self.downsamples.append(
                conv(
                    # in_channels[idx],
                    # in_channels[idx],
                    channels_dict[layer_name_downsample][0],
                    channels_dict[layer_name_downsample][1],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    # in_channels[idx] * 2,
                    # in_channels[idx + 1],
                    channels_dict[layer_name_bu][0],
                    channels_dict[layer_name_bu][1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        out_convs_in_channel = [channels_dict['top_down_blocks1'][1],
                                channels_dict['bottom_up_blocks0'][1],
                                channels_dict['bottom_up_blocks1'][1]]
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    # in_channels[i],
                    # out_channels,
                    out_convs_in_channel[i],
                    new_out_channel,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN_tfs features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)