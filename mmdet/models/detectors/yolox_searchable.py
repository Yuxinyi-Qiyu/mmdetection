# Copyright (c) OpenMMLab. All rights reserved.
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from ..builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX
from ..algorithm.searchbase import SearchBase

# from .single_stage import SingleStageDetector
# from .kd_loss import *

@DETECTORS.register_module()
class YOLOX_Searchable(SearchBase, YOLOX):
    def __init__(self,
                 *args,
                 search_space=None,
                 bn_training_mode=True,
                 num_sample_training=4,
                 divisor=4,
                 retraining=False,
                 **kwargs
                 ):
        YOLOX.__init__(self, *args, **kwargs)
        SearchBase.__init__(self, bn_training_mode=bn_training_mode, num_sample_training=num_sample_training, divisor=divisor, retraining=retraining)

        self._random_size_interval = self._random_size_interval * self.num_sample_training
        self.search_space = search_space
        if self.search_space:
            self.widen_factor_backbone_range = search_space['widen_factor_backbone_range']
            self.deepen_factor_backbone_range = search_space['deepen_factor_backbone_range']
            self.widen_factor_neck_range = search_space['widen_factor_neck_range']
            self.widen_factor_head_range = search_space['widen_factor_head_range']

    def sample_arch(self, mode='ramdom'): # ?
        assert mode in ('max', 'min', 'random')
        arch = {}
        if mode in ('max', 'min'):
            fn = eval(mode)
            arch['widen_factor_backbone'] = tuple([fn(self.backbone_widen_factor_range)] * 5)
            arch['deepen_factor_backbone'] = tuple([fn(self.backbone_deepen_factor_range)] * 4)
            arch['widen_factor_neck'] = tuple([fn(self.neck_widen_factor_range)] * 8)
            arch['widen_factor_head'] = max(self.head_widen_factor_range)
        elif mode == 'random':
            arch['widen_factor_backbone'] = tuple(random.choices(self.backbone_widen_factor_range, k=5))
            arch['deepen_factor_backbone'] = tuple(random.choices(self.backbone_deepen_factor_range, k=4))
            arch['widen_factor_neck'] = tuple(random.choices(self.neck_widen_factor_range, k=8))
            arch['widen_factor_head'] = random.choice(self.head_widen_factor_range)
        else:
            raise NotImplementedError
        return arch

    def set_arch(self, arch_dict):
        self.backbone.set_arch(arch_dict) # divisor=self.divisor
        self.neck.set_arch(arch_dict) # divisor=self.divisor
        self.bbox_head.set_arch(arch_dict) # divisor=self.divisor

    # def extract_feat(self, img):
    #     """Directly extract features from the backbone+neck."""
    #     x = self.backbone(img) # len(x) 3
    #     input_neck = []
    #     for stage_out in x:
    #         input_neck.append(stage_out)
    #     x = self.neck(input_neck)
    #     # x = self.neck([x[0],x[1],x[2]])
    #     return x #tuple (tensor) len=3
    #
    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None):
    #     """
    #     Args:
    #         img (Tensor): Input images of shape (N, C, H, W).
    #             Typically these should be mean centered and std scaled.
    #         img_metas (list[dict]): A List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             :class:`mmdet.datasets.pipelines.Collect`.
    #         gt_bboxes (list[Tensor]): Each item are the truth boxes for each
    #             image in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): Class indices corresponding to each box
    #         gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
    #             boxes can be ignored when computing the loss.
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     # Multi-scale training
    #     img, gt_bboxes = self._preprocess(img, gt_bboxes)
    #
    #     losses = dict()
    #     if not isinstance(self.archs, list): # not sandwich
    #         self.archs = [self.arch]
    #
    #     for idx, arch in enumerate(self.archs):
    #         if self.search_backbone or self.search_neck:
    #             self.set_arch(arch)
    #
    #         # x = self.extract_feat(img)
    #         x = self.backbone(img)
    #         x = self.neck(x)
    #
    #         if len(self.archs) > 1 and self.inplace: # inplace distill
    #             if idx == 0: # 最大的子网
    #                 teacher_feat = x
    #             else:
    #                 kd_feat_loss = 0
    #                 student_feat = x
    #                 for i in range(len(student_feat)):
    #                     kd_feat_loss += self.kd_loss(student_feat[i], teacher_feat[i].detach(), i) * self.kd_weight
    #
    #                 losses.update({'kd_feat_loss_{}'.format(idx): kd_feat_loss})
    #
    #         head_loss = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
    #                                           gt_labels, gt_bboxes_ignore)
    #
    #         losses.update({'loss_cls_{}'.format(idx): head_loss['loss_cls']})
    #         losses.update({'loss_bbox_{}'.format(idx): head_loss['loss_bbox']})
    #         losses.update({'loss_obj_{}'.format(idx): head_loss['loss_obj']})
    #
    #     # random resizing
    #     if (self._progress_in_iter + 1) % self._random_size_interval == 0:
    #         self._input_size = self._random_resize()
    #     self._progress_in_iter += 1
    #
    #     self.archs = None
    #
    #     return losses
    #
    # def _preprocess(self, img, gt_bboxes):
    #     scale_y = self._input_size[0] / self._default_input_size[0]
    #     scale_x = self._input_size[1] / self._default_input_size[1]
    #     if scale_x != 1 or scale_y != 1:
    #         img = F.interpolate(
    #             img,
    #             size=self._input_size,
    #             mode='bilinear',
    #             align_corners=False)
    #         for gt_bbox in gt_bboxes:
    #             gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
    #             gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
    #     return img, gt_bboxes
    #
    # def _random_resize(self):
    #     tensor = torch.LongTensor(2).cuda()
    #
    #     if self.rank == 0:
    #         size = random.randint(*self._random_size_range)
    #         aspect_ratio = float(
    #             self._default_input_size[1]) / self._default_input_size[0]
    #         size = (self._size_multiplier * size,
    #                 self._size_multiplier * int(aspect_ratio * size))
    #         tensor[0] = size[0]
    #         tensor[1] = size[1]
    #
    #     if self.world_size > 1:
    #         dist.barrier()
    #         dist.broadcast(tensor, 0)
    #
    #     input_size = (tensor[0].item(), tensor[1].item())
    #     return input_size
