# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .usconv import sepc_conv, USsepc_conv, USConv2d, USLinear, USBatchNorm2d
from ..utils import CSPLayer


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4, # 3*4
            out_channels, # 24
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        # print("_____________")
        # print(x.size()) #torch.Size([8, 12, 320, 320])
        return self.conv(x)


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@BACKBONES.register_module()
class CSPDarknet_Searchable(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp

    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=[1.0, 1.0, 1.0, 1.0],
                 widen_factor=[1.0, 1.0, 1.0, 1.0, 1.0],
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor # todo
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor[0]),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor[i])
            out_channels = int(out_channels * widen_factor[i + 1])
            num_blocks = max(round(num_blocks * deepen_factor[i]), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

        # print('model')
        # for i, layer_name in enumerate(self.layers):
        #     layer = getattr(self, layer_name)
        #     print(layer_name)
        #     print(layer)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet_Searchable, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        # print('train backbone')
        outs = []
        arch_setting = self.arch_settings['P5']
        stage = 0
        # [[64, 128, 3, True, False], [128, 256, 9, True, False],
        #  [256, 512, 9, True, False], [512, 1024, 3, False, True]]
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            # print(layer)

            if layer_name == 'stem':
                x = layer(x)
                continue

            _, _, num_blocks, add_identity, use_spp = arch_setting[stage]

            # conv
            # print("layer[0]")
            # print(layer)
            x = layer[0](x)
            # print(layer[0])
            # print("fin!")

            # spp
            use_spp_x = 1 if use_spp else 0
            if use_spp:
                x = layer[1](x)
                # print(layer[1])

            # csp layer todo:如何直接调用csp layer的forward函数
            num_blocks = max(round(num_blocks * self.deepen_factor[i - 1]), 1)
            # print("num_blocks"+str(num_blocks))
            # print(layer[1 + use_spp_x])
            x_short = layer[1 + use_spp_x].short_conv(x) # todo name
            x_main = layer[1 + use_spp_x].main_conv(x)

            darknetbottleneck = layer[1 + use_spp_x].blocks  # Sequential
            for block_num in range(num_blocks):
                identity = x_main
                # print(darknetbottleneck[block_num])
                out = darknetbottleneck[block_num].conv1(x_main)
                out = darknetbottleneck[block_num].conv2(out)

                if add_identity:  # 是否有shorcut
                    out = out + identity

            x = torch.cat((x_main, x_short), dim=1)
            x = layer[1 + use_spp_x].final_conv(x)
            # print("csp_fin!")
            if i in self.out_indices:
                outs.append(x)
            stage = stage + 1

        return tuple(outs)

    def set_arch(self, arch, **kwargs):
        # base_channel = 64
        # base_channel = arch['base_c']  # 修改base channel
        # factor = [1, 2, 4 ,8, 16] # 每个stage的in_channel对应basechannel的倍数
        widen_factor = arch['widen_factor']
        # base_channel = max(int(self.arch_settings['P5'][0][0] * widen_factor // 16 * 16), 16)#todo
        # print("base_channel:"+str(base_channel))
        deepen_factor = arch['deepen_factor']
        self.widen_factor = widen_factor
        # deepen_factor = 1
        self.deepen_factor = deepen_factor
        arch_setting = self.arch_settings['P5']
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            if layer_name == "stem":
                channel = int(arch_setting[0][0] * widen_factor[0] // 16 * 16)
                # todo
                if channel == 0:
                    channel = 16
                layer.conv.conv.out_channels = channel
                layer.conv.bn.num_features = channel
                continue

            in_channels, out_channels, num_blocks, _, use_spp = arch_setting[i - 1] # todo 改成从arch setting里取出in/out channel * widen——factor
            num_blocks = max(round(num_blocks * deepen_factor[i - 1]), 1) # deepfactor
            use_spp_x = 1 if use_spp else 0
            in_channel = int(in_channels * widen_factor[i - 1] // 16 * 16)
            out_channel = int(out_channels * widen_factor[i] // 16 * 16)
            # todo 太丑了
            if in_channel == 0:
                in_channel = 16
            if out_channel == 0:
                out_channel = 16
            # convmodule
            # in_channel = base_channel * factor[i - 1]
            # out_channel = base_channel * factor[i]
            layer[0].conv.in_channels, layer[0].conv.out_channels = in_channel, out_channel
            layer[0].bn.num_features = out_channel
            if use_spp:
                # sppbottleneck  ?conv2_channels = mid_channels * (len(kernel_sizes) + 1)
                in_channel = out_channel
                mid_channel = in_channel // 2
                out_channel = out_channel
                layer[1].conv1.conv.in_channels, layer[1].conv1.conv.out_channels = in_channel, mid_channel
                layer[1].conv1.bn.num_features = mid_channel
                layer[1].conv2.conv.in_channels, layer[1].conv2.conv.out_channels = mid_channel * 4, out_channel
                layer[1].conv2.bn.num_features = out_channel
            # CSPlayer mid_channels = int(out_channels * expand_ratio)
            # num_blocks = max(round(num_blocks * deepen_factor), 1)
            in_channel = out_channel
            mid_channel = int(out_channel * 0.5)
            layer[1 + use_spp_x].main_conv.conv.in_channels, layer[1 + use_spp_x].main_conv.conv.out_channels = in_channel, mid_channel
            layer[1 + use_spp_x].main_conv.bn.num_features = mid_channel
            layer[1 + use_spp_x].short_conv.conv.in_channels, layer[1 + use_spp_x].short_conv.conv.out_channels = in_channel, mid_channel
            layer[1 + use_spp_x].short_conv.bn.num_features = mid_channel
            layer[1 + use_spp_x].final_conv.conv.in_channels, layer[1 + use_spp_x].final_conv.conv.out_channels = mid_channel * 2, out_channel
            layer[1 + use_spp_x].final_conv.bn.num_features = out_channel
            # DarknetBottleneck
            darknetbottleneck = layer[1 + use_spp_x].blocks  # Sequential
            for block_num in range(num_blocks):
                hidden_channel = mid_channel
                darknetbottleneck[block_num].conv1.conv.in_channels, darknetbottleneck[block_num].conv1.conv.out_channels = mid_channel, hidden_channel
                darknetbottleneck[block_num].conv1.bn.num_features = hidden_channel
                darknetbottleneck[block_num].conv2.conv.in_channels, darknetbottleneck[block_num].conv2.conv.out_channels = hidden_channel, mid_channel
                darknetbottleneck[block_num].conv2.bn.num_features = mid_channel

        # print("<<<<<<<<<<<<<<<<<<<<<<")
        # print("the model")
        # for i, layer_name in enumerate(self.layers):
        #     layer = getattr(self, layer_name)
        #     print("i=" + str(i) + ":" + str(layer))

        '''
            # layer.conv.conv.in_channels = 16
            arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]]
    }
    in_channels, out_channels, num_blocks, add_identity,
                use_spp
    factor = [1, 2, 4 ,8, 16]'''