import warnings

import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .two_stage import TwoStageDetector
from einops import rearrange, reduce
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..utils import USConv2d, USBatchNorm2d

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.epsilon=1e-4
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(x)
        x = x / (torch.sum(x, dim=0) + self.epsilon)
        return x

class DML(nn.Module):
    def __init__(self):
        super(DML,self).__init__()
        self.norm1 = Norm()
        self.norm2 = Norm()
    def forward(self, x1, x2, i):
        x1 = max_pooling_layer(x1)
        x2 = max_pooling_layer(x2)
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        loss = F.kl_div(x1,x2,
                        reduction='batchmean')
        return loss

class DL2(nn.Module):
    def __init__(self, softmax=False):
        super(DL2,self).__init__()
        self.softmax = softmax
        if self.softmax:
            self.norm = Norm()

    def forward(self, x1, x2, i):
        x1 = max_pooling_layer(x1)
        x2 = max_pooling_layer(x2)
        if self.softmax:
            x1 = self.norm(x1)
            x2 = self.norm(x2)
        diff = (x1 - x2) **2
        loss = torch.sum(diff) ** 0.5
        return loss



def max_pooling_layer(x):
    return reduce(x, 'b c h w -> b 1 h w', 'max')


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True,  downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = USConv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.reduct = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        self.g.in_channels = self.in_channels
        self.theta.in_channels = self.in_channels
        self.phi.in_channels = self.in_channels
        self.reduct.in_channels = self.in_channels
        if self.sub_sample:
            self.g[0].in_channels = self.in_channels
            self.phi[0].in_channels = self.in_channels

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        x = self.reduct(x)
        z = W_y + x

        return z

class NonLocalBlockLoss(nn.Module):
    def __init__(self, in_channels=256, out_channels=64):
        super(NonLocalBlockLoss,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels, downsample_stride=8),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels, downsample_stride=4),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels),

            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels, downsample_stride=8),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels, downsample_stride=4),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels),
                NonLocalBlockND(in_channels=in_channels, inter_channels=out_channels),

            ]
        )

        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        ] * 5)
        self.norm_s = nn.ModuleList([Norm()]*5)
        self.norm_t = nn.ModuleList([Norm()]*5)


    def forward(self, x_s, x_t, i):
        self.student_non_local[i].in_channels = x_s.shape[1]
        s_relation = self.student_non_local[i](x_s)
        t_relation = self.teacher_non_local[i](x_t)
        s_relation = self.norm_s[i](self.non_local_adaptation[i](s_relation))
        t_relation = self.norm_t[i](t_relation)
        # loss = torch.dist(s_relation, t_relation, p=2)
        diff = (s_relation - t_relation) ** 2
        loss = torch.sum(diff) ** 0.5

        return loss
