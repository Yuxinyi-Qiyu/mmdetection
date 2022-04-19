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
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        # in_channels=[128, 256, 512],
        # in_channels=[256, 512, 1024],
        #         out_channels=128,
        #         num_csp_blocks=1
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2, # 256*2
                    in_channels[idx - 1], #256
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
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
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
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def set_arch(self, arch, **kwargs):
        widen_factor = arch['widen_factor']
        # in_channels = self.in_channels  todo 不知道为什么这么赋值之后，在下一次进入的时候self就变了
        in_channels = [256, 512, 1024]
        out_channels = self.out_channels
        # base_channel = int(in_channels[0] * widen_factor // 16 * 16) # todo：改成以in_channel为基准的
        # base_channel = max(int(64 * widen_factor // 16 * 16), 16)
        # base_channel = base_channel * 4
        # print("in_channels")
        # print(in_channels)
        for i in range(len(in_channels)):
            in_channels[i] = int(in_channels[i] * widen_factor[i + 2] // 16 * 16)
            if in_channels[i] == 0:
                in_channels[i] = 16

        # print("base_channel:"+str(base_channel))
        # print("in_channels")
        # print(in_channels)
        # factor = [4 ,8, 16] # 每个stage的in_channel对应basechannel的倍数
        # factor = [1 ,2, 4] # 每个stage的in_channel对应basechannel的倍数
        # widen_factor = 0.5
        expansion_ratio = 0.5 # todo 搞清楚这是个啥

        # for i in range(len(factor)):
        #     # factor[i] = int(factor[i] * widen_factor)
        #     factor[i] = int(factor[i] * widen_factor)

        for idx in range(len(self.in_channels) - 1):
            # reduce_layers
            # in_channel = base_channel * factor[len(factor) - 1 - idx]
            # out_channel = base_channel * factor[len(factor) - 1 - idx - 1]
            in_channel = in_channels[len(in_channels) - 1 - idx]
            out_channel = in_channels[len(in_channels) - 1 - idx - 1]
            self.reduce_layers[idx].conv.in_channels, self.reduce_layers[idx].conv.out_channels = in_channel, out_channel
            self.reduce_layers[idx].bn.num_features = out_channel
            # top_down_blocks
            mid_channel = int(out_channel * expansion_ratio)
            # self.top_down_blocks[idx].main_conv.conv.in_channels, self.top_down_blocks[idx].main_conv.conv.out_channels = in_channel, mid_channel
            self.top_down_blocks[idx].main_conv.conv.in_channels, self.top_down_blocks[idx].main_conv.conv.out_channels = out_channel * 2, mid_channel
            self.top_down_blocks[idx].main_conv.bn.num_features = mid_channel
            # self.top_down_blocks[idx].short_conv.conv.in_channels, self.top_down_blocks[idx].short_conv.conv.out_channels = in_channel, mid_channel
            self.top_down_blocks[idx].short_conv.conv.in_channels, self.top_down_blocks[idx].short_conv.conv.out_channels = out_channel * 2, mid_channel
            self.top_down_blocks[idx].short_conv.bn.num_features = mid_channel
            self.top_down_blocks[idx].final_conv.conv.in_channels, self.top_down_blocks[idx].final_conv.conv.out_channels = 2 * mid_channel, out_channel
            self.top_down_blocks[idx].final_conv.bn.num_features = out_channel
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

        for idx in range(len(self.in_channels) - 1):
            # downsamples
            # channel = base_channel * factor[idx] # 128
            channel = in_channels[idx]
            # print(self.downsamples)
            # print(self.downsamples[0].conv)
            self.downsamples[idx].conv.in_channels, self.downsamples[idx].conv.out_channels = channel, channel
            self.downsamples[idx].bn.num_features = channel
            # bottom_up_blocks
            in_channel = channel * 2 # 256
            # out_channel = base_channel * factor[idx + 1] # 256
            out_channel = in_channels[idx + 1]
            mid_channel = int(out_channel * 0.5) # 128
            self.bottom_up_blocks[idx].main_conv.conv.in_channels, self.bottom_up_blocks[
                idx].main_conv.conv.out_channels = in_channel, mid_channel
            self.bottom_up_blocks[idx].main_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].short_conv.conv.in_channels, self.bottom_up_blocks[
                idx].short_conv.conv.out_channels = in_channel, mid_channel
            self.bottom_up_blocks[idx].short_conv.bn.num_features = mid_channel
            self.bottom_up_blocks[idx].final_conv.conv.in_channels, self.bottom_up_blocks[
                idx].final_conv.conv.out_channels = 2 * mid_channel, out_channel
            self.bottom_up_blocks[idx].final_conv.bn.num_features = out_channel
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
        for idx in range(len(in_channels)):
            # in_channel = base_channel * factor[idx]
            in_channel = in_channels[idx]
            out_channel = out_channels #todo 现在都是固定的，以后改成可变的
            self.out_convs[idx].conv.in_channels = in_channel
            # self.out_convs[idx].conv.in_channels, self.out_convs[idx].conv.out_channels = channel, out_channel
            # self.out_convs[idx].bn.num_features = out_channel
            # self.out_convs[idx].bn.bn.num_features = out_channel

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
        # print("!!!!!!!!neck")
        # print("!!!!!!!!top-down path")
        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            # print("!!!!!!!!reduce_layers:idx"+str(idx))
            # print(feat_heigh.size()) # torch.Size([8, 384, 20, 20])
            # print(len(self.in_channels) - 1 - idx)
            # print(self.reduce_layers[len(self.in_channels) - 1 - idx]) # 0
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh
            # print("!!!!!!!!fin")

            upsample_feat = self.upsample(feat_heigh)
            # print("!!!!!!!!top_down_blocks:idx"+str(idx))
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)
            # print("!!!!!!!!fin")


        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            # print("!!!!!!!!downsamples:idx"+str(idx))
            downsample_feat = self.downsamples[idx](feat_low)
            # print("!!!!!!!!fin")
            # print("!!!!!!!!bottom_up_blocks:idx"+str(idx))
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)
            # print("!!!!!!!!fin")


        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
            # print(outs[idx].size())

        return tuple(outs)

