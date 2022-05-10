# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import CSPLayer


@NECKS.register_module("YOLOXPAFPN_Searchable")
class YOLOXPAFPN_Searchable(BaseModule):
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
                 widen_factor=[0.5, 0.5, 0.5, 0.5],
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
        super(YOLOXPAFPN_Searchable, self).__init__(init_cfg)
        self.widen_factor = widen_factor
        self.widen_factor_out = widen_factor_out
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = [512, 256, 256, 512]
        self.base_channels_backbone = [256, 512, 1024]
        self.base_out_channels = 256
        new_channels_reduce = []
        for i in range(2):
            new_channels_reduce.append(int(self.base_channels[i] * self.widen_factor[i]))
        new_channels_bu = []
        for i in range(2):
            new_channels_bu.append(int(self.base_channels[i + 2] * self.widen_factor[i + 2]))
        new_out_channel = int(self.base_out_channels * self.widen_factor_out)
        self.new_out_channel = new_out_channel
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        # print(new_channels_reduce)
        # print(new_channels_bu)
        self.upsample = nn.Upsample(**upsample_cfg)
        self.conv1x1 = nn.ModuleList()
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        # for idx in range(len(in_channels) - 1, -1, -1):
        #     self.conv1x1.append(
        #         ConvModule(
        #             in_channels[idx],
        #             new_channels[idx],
        #             1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))

        # build top-down blocks
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    # new_channels[idx],
                    # new_channels[idx - 1],
                    in_channels[idx], # 2 1
                    new_channels_reduce[2 - idx], # 0 1
                    # in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    # new_channels[idx - 1] * 2, # 256*2
                    # new_channels[idx - 1], #256
                    # in_channels[idx - 1] * 2,
                    (new_channels_reduce[2 - idx] + in_channels[idx - 1]),
                    in_channels[idx - 1],
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
            self.downsamples.append(
                conv(
                    # new_channels[idx],
                    # new_channels[idx],
                    in_channels[idx], # 0, 1
                    new_channels_bu[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            # print("bottom_up_blocks")
            # print(new_channels_reduce[1 - idx] + new_channels_bu[idx])
            self.bottom_up_blocks.append(
                CSPLayer(
                    (new_channels_reduce[1 - idx] + new_channels_bu[idx]),
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    new_out_channel,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        
    def set_arch(self, arch, **kwargs):
        widen_factor_backbone = arch['widen_factor_backbone']
        in_channels = []
        for i in range(len(self.in_channels)):
            in_channels.append(int(self.base_channels_backbone[i] * widen_factor_backbone[i + 2]))

        widen_factor_neck = arch['widen_factor_neck']
        new_channels_reduce = []
        for i in range(2):
            new_channels_reduce.append(int(self.base_channels[i] * widen_factor_neck[i]))
        new_channels_bu = []
        for i in range(2):
            new_channels_bu.append(int(self.base_channels[i + 2] * widen_factor_neck[i + 2]))

        widen_factor_out_neck = arch['widen_factor_neck_out']
        new_out_channel = int(self.base_out_channels * widen_factor_out_neck)
        out_channels = new_out_channel

        expansion_ratio = 0.5 # todo 搞清楚这是个啥

        # for i in range(len(self.in_channels)):
        #     self.conv1x1[len(self.in_channels) - 1 - i].conv.in_channels, self.conv1x1[len(self.in_channels) - 1 - i].conv.out_channels = in_channels_backbone[i], in_channels_neck[i]
        #     self.conv1x1[len(self.in_channels) - 1 - i].bn.num_features = in_channels_neck[i]

        # in_channels = in_channels_neck

        for idx in range(len(self.in_channels) - 1): # 0, 1
            # reduce_layers
            in_channel = in_channels[len(self.in_channels) - 1 - idx]
            out_channel_reduce = new_channels_reduce[idx]
            out_channel_csp = in_channels[len(self.in_channels) - 2 - idx]
            self.reduce_layers[idx].conv.in_channels, self.reduce_layers[idx].conv.out_channels = in_channel, out_channel_reduce
            self.reduce_layers[idx].bn.num_features = out_channel_reduce
            # top_down_blocks
            mid_channel = int(out_channel_csp * expansion_ratio)
            # self.top_down_blocks[idx].main_conv.conv.in_channels, self.top_down_blocks[idx].main_conv.conv.out_channels = in_channel, mid_channel
            self.top_down_blocks[idx].main_conv.conv.in_channels, self.top_down_blocks[idx].main_conv.conv.out_channels = (out_channel_reduce + out_channel_csp), mid_channel
            self.top_down_blocks[idx].main_conv.bn.num_features = mid_channel
            # self.top_down_blocks[idx].short_conv.conv.in_channels, self.top_down_blocks[idx].short_conv.conv.out_channels = in_channel, mid_channel
            self.top_down_blocks[idx].short_conv.conv.in_channels, self.top_down_blocks[idx].short_conv.conv.out_channels = (out_channel_reduce + out_channel_csp), mid_channel
            self.top_down_blocks[idx].short_conv.bn.num_features = mid_channel
            self.top_down_blocks[idx].final_conv.conv.in_channels, self.top_down_blocks[idx].final_conv.conv.out_channels = 2 * mid_channel, out_channel_csp
            self.top_down_blocks[idx].final_conv.bn.num_features = out_channel_csp
            darknetbottleneck = self.top_down_blocks[idx].blocks  # Sequential
            num_blocks = 1
            for block_num in range(num_blocks):
                hidden_channel = mid_channel
                darknetbottleneck[block_num].conv1.conv.in_channels, darknetbottleneck[
                    block_num].conv1.conv.out_channels = mid_channel, hidden_channel
                darknetbottleneck[block_num].conv1.bn.num_features = hidden_channel
                darknetbottleneck[block_num].conv2.conv.in_channels, darknetbottleneck[
                    block_num].conv2.conv.out_channels = hidden_channel, mid_channel
                darknetbottleneck[block_num].conv2.bn.num_features = mid_channel

        for idx in range(len(self.in_channels) - 1): # 0, 1
            in_channel = in_channels[idx]
            out_channel_bu = new_channels_bu[idx]
            out_channel_csp = in_channels[idx + 1]
            in_channnel_csp = out_channel_bu + new_channels_reduce[1 - idx]
            # downsamples
            self.downsamples[idx].conv.in_channels, self.downsamples[idx].conv.out_channels = in_channel, out_channel_bu
            self.downsamples[idx].bn.num_features = out_channel_bu
            # bottom_up_blocks
            mid_channel = int(out_channel_csp * expansion_ratio) # 128
            self.bottom_up_blocks[idx].main_conv.conv.in_channels, self.bottom_up_blocks[
                idx].main_conv.conv.out_channels = in_channnel_csp, mid_channel
            self.bottom_up_blocks[idx].main_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].short_conv.conv.in_channels, self.bottom_up_blocks[
                idx].short_conv.conv.out_channels = in_channnel_csp, mid_channel
            self.bottom_up_blocks[idx].short_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].final_conv.conv.in_channels, self.bottom_up_blocks[
                idx].final_conv.conv.out_channels = 2 * mid_channel, out_channel_csp
            self.bottom_up_blocks[idx].final_conv.bn.num_features = out_channel_csp
            darknetbottleneck = self.bottom_up_blocks[idx].blocks  # Sequential
            num_blocks = 1
            for block_num in range(num_blocks):
                hidden_channel = mid_channel # 128
                darknetbottleneck[block_num].conv1.conv.in_channels, darknetbottleneck[
                    block_num].conv1.conv.out_channels = mid_channel, hidden_channel
                darknetbottleneck[block_num].conv1.bn.num_features = hidden_channel
                darknetbottleneck[block_num].conv2.conv.in_channels, darknetbottleneck[
                    block_num].conv2.conv.out_channels = hidden_channel, mid_channel
                darknetbottleneck[block_num].conv2.bn.num_features = mid_channel
        # upsample?
        # out_convs
        for idx in range(len(self.in_channels)):
            # in_channel = base_channel * factor[idx]
            in_channel = in_channels[idx]
            out_channel = out_channels #todo 现在都是固定的，以后改成可变的
            self.out_convs[idx].conv.in_channels = in_channel
            self.out_convs[idx].conv.out_channels = out_channel
            self.out_convs[idx].bn.num_features = out_channel
            # self.out_convs[idx].conv.in_channels, self.out_convs[idx].conv.out_channels = channel, out_channel
            # self.out_convs[idx].bn.num_features = out_channel

        # print("<<<<<<<<<<<<<<<<<<<<")
        # print("self.reduce_layers")
        # print(self.reduce_layers)
        # print("self.top_down_blocks")
        # print(self.top_down_blocks)
        # print("self.downsamples")
        # print(self.downsamples)
        # print("self.bottom_up_blocks")
        # print(self.bottom_up_blocks)
        # print("self.out_convs")
        # print(self.out_convs)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # for idx in range(len(self.in_channels) - 1, -1, -1):
        #     inputs[idx] = self.conv1x1[len(self.in_channels) - 1 - idx](inputs[idx])

        # print("top_down")
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

        # print("bottom_up")
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            # print("!!!!!!!!downsamples:idx"+str(idx))
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # print("out_conv")
        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
            # print(outs[idx].size())
        # print("finish_neck")
        return tuple(outs)

