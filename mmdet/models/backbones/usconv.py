# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, CONV_LAYERS, NORM_LAYERS
from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2d, deform_conv2d, ModulatedDeformConv2d, modulated_deform_conv2d, SyncBatchNorm, \
    DeformConv2dPack, ModulatedDeformConv2dPack
from torch.nn.modules.utils import _pair
import torch
import math
from mmcv.cnn import build_conv_layer, build_norm_layer

@CONV_LAYERS.register_module('USConv2d')
class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise

    def forward(self, input): # 表示卷积呗
        # print("self.in_channels")
        # print(self.in_channels) # 768
        # print("self.out_channels")
        # print(self.out_channels) # 384
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        # 怎么取self weight（未见到定义  ：？可能是conv的性质？
        # 卷积这四个维度的意思 fin

        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        # print("usconv-fin")
        # print("~~~~~~"+str(y.shape)) # torch.Size([8, 24, 320, 320])
        # print("~~~~~~out"+str(self.out_channels))  #24
        # print("~~~~~~in"+str(self.in_channels)) #12
        return y


@CONV_LAYERS.register_module('SEPC_DCN')
class SEPC_DCN(DeformConv2d):
    def __init__(self, *args, **kwargs, ):
        super(SEPC_DCN, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups) + self.bias.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)
        return out


@CONV_LAYERS.register_module('USDCN')
class USDeformConv2dPack(DeformConv2dPack):

    def __init__(self, *args, **kwargs):
        super(USDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = USConv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)

    def forward(self, x):
        self.conv_offset.in_channels = self.in_channels
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, weight.clone(), self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)


@CONV_LAYERS.register_module('USDCNv2')
class USModulatedDeformConv2dPack(ModulatedDeformConv2dPack):

    def __init__(self, *args, **kwargs):
        super(USModulatedDeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = USConv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)

    def forward(self, x):
        self.conv_offset.in_channels = self.in_channels
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels].clone()
        else:
            bias = self.bias

        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = x.type_as(offset)
        weight = weight.type_as(x)
        bias = bias.type_as(x)
        return modulated_deform_conv2d(x, offset, mask, weight, bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@CONV_LAYERS.register_module('DCNv216')
class ModulatedDeformConv2dPack16(ModulatedDeformConv2dPack):

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConv2dPack16, self).__init__(*args, **kwargs)

    def forward(self, x):

        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = x.type_as(offset)
        weight = self.weight.type_as(x)
        if self.bias is not None:
            bias = self.bias.type_as(x)
        else:
            bias = self.bias

        return modulated_deform_conv2d(x, offset, mask, weight, bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@CONV_LAYERS.register_module('FA_DCNv2')
class FA_DCNv2(ModulatedDeformConv2dPack):
    def __init__(self, *args, **kwargs):
        super(FA_DCNv2, self).__init__(*args, **kwargs)

        channels_ = self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset(input[1])
        input = input[0]
        o1, o2, mask = torch.chunk(out, 3,
                                   dim=1)  # each has self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] channels
        offset = torch.cat((o1, o2), dim=1)  # x, y [0-8]: the first group,
        mask = torch.sigmoid(mask)

        input = input.type_as(offset)
        weight = self.weight.type_as(input)
        bias = self.bias.type_as(input)
        return modulated_deform_conv2d(input, offset, mask, weight, bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@CONV_LAYERS.register_module('USFA_DCNv2')
class USFA_DCNv2(ModulatedDeformConv2dPack):
    def __init__(self, *args, **kwargs):
        super(USFA_DCNv2, self).__init__(*args, **kwargs)

        channels_ = self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = USConv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride,
                                    padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        self.conv_offset.in_channels = self.in_channels
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels].clone()
        else:
            bias = self.bias

        out = self.conv_offset(input[1])
        input = input[0]
        o1, o2, mask = torch.chunk(out, 3,
                                   dim=1)  # each has self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] channels
        offset = torch.cat((o1, o2), dim=1)  # x, y [0-8]: the first group,
        mask = torch.sigmoid(mask)

        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)
        return modulated_deform_conv2d(input, offset, mask, weight.clone(), bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@CONV_LAYERS.register_module('FA_DCNv2_FP')
class FA_DCNv2_FP(nn.Conv2d):
    def __init__(self, *args, deform_groups=1, **kwargs):
        super(FA_DCNv2_FP, self).__init__(*args, **kwargs)
        self.deform_groups = deform_groups
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                              padding=self.padding, bias=True)
        channels_ = self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        if isinstance(input, list):
            input = input[0]
        out = self.conv_offset(input)
        # out = self._conv_forward(input, self.weight)
        out = self.conv(input)
        return out

@NORM_LAYERS.register_module('USBN2d')
class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 group=None,
                 stats_mode='default',
                 fea_range=[64, 384]):
        super(USBatchNorm2d, self).__init__(
            num_features=num_features, affine=True)
        # self.num_features_max = num_features

        # self.bn = nn.BatchNorm2d(self.num_features_max, affine=False)

        self.training = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        # print("input.size()")
        # print(input.size())
        # print("self.num_features")
        # print(self.num_features)
        y = nn.functional.batch_norm(
            input,
            self.running_mean[:self.num_features],
            self.running_var[:self.num_features],
            weight[:self.num_features],
            bias[:self.num_features],
            self.training,
            self.momentum,
            self.eps)

        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features

    def forward(self, input):
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class sepc_conv(DeformConv2d):
    def __init__(self, *args, part_deform=True, **kwargs, ):
        super(sepc_conv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        if part_deform:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 2 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                bias=True)
            self.init_offset()
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1
        self.syncbn = SyncBatchNorm(self.out_channels)

    # def reset_parameters(self):
    #     n = self.in_channels
    #     for k in self.kernel_size:
    #         n *= k
    #     stdv = 1. / math.sqrt(n) * math.sqrt(2)
    #     self.weight.data.uniform_(-stdv, stdv)
    # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')

    def init_offset(self):

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        if i < self.start_level or not self.part_deform:
            out = torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                             dilation=self.dilation, groups=self.groups)
            out = self.syncbn(out)
            return out

        offset = self.conv_offset(x)
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups) + self.bias.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)
        out = self.syncbn(out)

        return out


class mdsepc_conv(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs, ):
        super(mdsepc_conv, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()
        self.start_level = 1

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        if i < self.start_level:
            return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                              dilation=self.dilation, groups=self.groups)

        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class USsepc_conv(sepc_conv):
    def __init__(self, *args, **kwargs, ):
        super(USsepc_conv, self).__init__(*args, **kwargs)
        if self.part_deform:
            self.conv_offset = USConv2d(
                self.in_channels,
                self.deform_groups * 2 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                bias=True)

    def forward(self, i, x):
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding,
                                              dilation=self.dilation, groups=self.groups)

        self.conv_offset.in_channels = self.in_channels
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, weight.clone(), self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups) + bias.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)


class usmdsepc_conv(mdsepc_conv):
    def __init__(self, *args, **kwargs, ):
        super(usmdsepc_conv, self).__init__(*args, **kwargs)
        self.conv_offset = USConv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)

    def forward(self, i, x):
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        if i < self.start_level:
            return torch.nn.functional.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding,
                                              dilation=self.dilation, groups=self.groups)

        self.conv_offset.in_channels = self.in_channels
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, weight.clone(), bias.clone(),
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class FeatureAlign_V2(ModulatedDeformConv2d):
    def __init__(self, *args, fuse_num=2, **kwargs, ):
        super(FeatureAlign_V2, self).__init__(*args, **kwargs)
        self.fuse_num = fuse_num
        self.fuse_offset = nn.Conv2d(self.in_channels * self.fuse_num, self.out_channels, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()
        self.start_level = 1

    def init_offset(self):

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        target_size = x[0].shape[2:]

        cat_feat = []
        cat_feat.append(x[0])
        feat = x[1]
        if feat.shape[2:] > target_size:
            feat = F.max_pool2d(feat.clone(), 2, stride=2, ceil_mode=False)
        else:
            feat = F.interpolate(feat.clone(), target_size, mode='bilinear', align_corners=False)
        cat_feat.append(feat)
        fuse_offset = self.fuse_offset(torch.cat(cat_feat, dim=1))

        out = self.conv_offset(fuse_offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(feat, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups) + x[0].clone()

