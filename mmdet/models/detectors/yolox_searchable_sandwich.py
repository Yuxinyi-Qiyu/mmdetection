# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from ..builder import DETECTORS
from .single_stage import SingleStageDetector

def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    if tensor_a.shape != tensor_b:
        tensor_a = max_pooling_layer(tensor_a)
        tensor_b = max_pooling_layer(tensor_b)
    diff = (tensor_a - tensor_b) ** 2
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff

def max_pooling_layer(x):
    return reduce(x, 'b c h w -> b 1 h w', 'max')

@DETECTORS.register_module()
class YOLOX_Searchable_Sandwich(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size.
            Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None,
                 search_backbone=True,
                 search_neck=True,
                 search_head=False,
                 sandwich=False,
                 inplace=False, # distill
                 kd_weight=1e-8,
                 ):
        super(YOLOX_Searchable_Sandwich, self).__init__(
            backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg,
        )

        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0
        self.search_backbone = search_backbone
        self.search_neck = search_neck
        self.search_head = search_head
        self.sandwich = sandwich
        self.inplace = inplace
        self.arch = None
        self.archs = None
        # 不同蒸馏对应的loss计算方法
        if self.inplace == 'L2':
            self.kd_loss = DL2()
        elif self.inplace == 'L2Softmax':
            self.kd_loss = DL2(softmax=True)
        elif self.inplace == 'DML':
            self.kd_loss = DML()
        elif self.inplace == 'NonLocal':
            self.kd_loss = NonLocalBlockLoss(self.out_channels, 64)
        # self.set_arch({'panas_arch': (3, 3, 1, 2, -1), 'panas_c': 112, 'panas_d': 4, 'cb_type': 0, 'cb_step': 2, 'head_step': 2})

    def set_archs(self, archs, **kwargs):
        self.archs = archs

    def set_arch(self, arch, **kwargs):
        self.arch = arch
        if self.search_backbone:
            self.backbone.set_arch(self.arch)
        if self.search_neck:
            self.neck.set_arch(self.arch)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""

        if self.search_backbone:
            x = self.backbone(img, self.cb_step) #!!
        else:
            x = self.backbone(img)
        if not isinstance(x[0], (list, tuple)):
            x = [x]

        if self.search_neck: # ?
            outs = []
            for backbone_out in x:
                out = self.neck(backbone_out)
                outs.append(out)
                return outs
        else:
            out = self.neck(x[-1])
            return out

        return x

        # if self.with_neck: # ?
        #     if self.training:
        #         outs = []
        #         for cb_out in x:
        #             out = self.neck(cb_out)
        #             outs.append(out)
        #         return outs
        #     else:
        #         out = self.neck(x[-1])
        #         return out
        # return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        losses = dict()
        if not isinstance(self.archs, list): # not sandwich
            self.archs = [self.arch]
        # if self.sandwich: # backbone结构一样时，节省时间，直接复用backbone feature
        #     backbone_feat = self.extract_backbone_feat(img)
        for idx, arch in enumerate(self.archs):
            if self.search_backbone or self.search_neck:
                self.set_arch(arch)

            # if self.sandwich: # 这里省略，因为backbone会变
            #     xs = self.extract_neck_feat(backbone_feat)
            # else:
            #     xs = self.extract_feat(img)

            xs = self.extract_feat(img) # ?

            if len(self.archs) > 1 and self.inplace: # inplace distill
                if idx == 0: # 最大的子网
                    teacher_feat = xs[-1]
                else:
                    kd_feat_loss = 0
                    student_feat = xs[-1]
                    for i in range(len(student_feat)):
                        kd_feat_loss += self.kd_loss(student_feat[i], teacher_feat[i].detach(), i) * self.kd_weight

                    losses.update({'kd_feat_loss_{}'.format(idx): kd_feat_loss})


        # img, gt_bboxes = self._preprocess(img, gt_bboxes)
        #
        # losses = super(YOLOX_Searchable, self).forward_train(img, img_metas, gt_bboxes,
        #                                           gt_labels, gt_bboxes_ignore)
        #
        # # random resizing
        # if (self._progress_in_iter + 1) % self._random_size_interval == 0:
        #     self._input_size = self._random_resize()
        # self._progress_in_iter += 1

        return losses

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self):
        tensor = torch.LongTensor(2).cuda()

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            aspect_ratio = float(
                self._default_input_size[1]) / self._default_input_size[0]
            size = (self._size_multiplier * size,
                    self._size_multiplier * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
