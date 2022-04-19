# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .two_stage import TwoStageDetector
from einops import rearrange, reduce
import torch.nn.functional as F
from .kd_loss import *

@DETECTORS.register_module()
class CBTwoStageDetector(TwoStageDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 search_backbone=False,
                 search_neck=False,
                 search_head=False,
                 sandwich=False,
                 inplace=False,
                 kd_weight=1e-8,
                 ):
        super(CBTwoStageDetector, self).__init__(
                backbone,
                neck,
                rpn_head,
                roi_head,
                train_cfg,
                test_cfg,
                pretrained,
                init_cfg,
        )

        self.cb_step = -1
        self.search_backbone = search_backbone
        self.search_neck = search_neck
        self.search_head = search_head
        self.inplace = inplace
        self.sandwich = sandwich
        if not self.training:
            self.sandwich = False
        self.arch = [1,1,2]
        self.archs = [self.arch]
        self.cascade = False
        self.out_channels = self.neck.out_channels
        if 'Cascade' in roi_head.type:
            self.cascade = True
        self.kd_weight = kd_weight
        # 不同蒸馏对应的loss计算方法
        if self.inplace == 'L2':
            self.kd_loss = DL2()
        elif self.inplace == 'L2Softmax':
            self.kd_loss = DL2(softmax=True)
        elif self.inplace == 'DML':
            self.kd_loss = DML()
        elif self.inplace == 'NonLocal':
            self.kd_loss = NonLocalBlockLoss(self.out_channels, 64)
        self.set_arch({'panas_arch': (3, 3, 1, 2, -1), 'panas_c': 112, 'panas_d': 4, 'cb_type': 0, 'cb_step': 2, 'head_step': 2})


    def set_archs(self, archs, **kwargs):
        self.archs = archs

    def set_arch(self, arch, **kwargs):
        self.arch = arch
        if (not self.training) and self.search_backbone:
            self.cb_step = arch['cb_step']
        if self.with_neck and self.search_neck:
            self.neck.set_arch(self.arch)
            if self.with_rpn:
                self.rpn_head.set_arch(self.arch)
            if self.with_roi_head:
                if self.cascade:
                    for roi_extractor, head in zip(self.roi_head.bbox_roi_extractor, self.roi_head.bbox_head):
                        roi_extractor.out_channels = self.arch['panas_c']
                        head.set_arch(self.arch)
                else:
                    self.roi_head.bbox_roi_extractor.out_channels = self.arch['panas_c']
                    self.roi_head.bbox_head.set_arch(self.arch)
                if hasattr(self.roi_head, 'mask_head'):
                    if self.cascade:
                        for roi_extractor, head in zip(self.roi_head.mask_roi_extractor, self.roi_head.mask_head):
                            roi_extractor.out_channels = self.arch['panas_c']
                            head.set_arch(self.arch)
                            # head.convs[0].conv.in_channels = self.arch['panas_c']
                    else:
                        self.roi_head.mask_roi_extractor.out_channels = self.arch['panas_c']
                        self.roi_head.mask_head.convs[0].conv.in_channels = self.arch['panas_c']
        if self.with_roi_head and self.search_head and (not self.training) and self.cascade:
            self.roi_head.num_stages = self.arch['head_step']


    def extract_backbone_feat(self, img):
        if (not self.training) and self.search_backbone:
            x = self.backbone(img, self.cb_step)
        else:
            x = self.backbone(img)
        if not isinstance(x[0], (list, tuple)):
            x = [x]
        return x

    def extract_neck_feat(self, x):
        if self.with_neck:
            if self.training:
                outs = []
                for cb_out in x:
                    out = self.neck(cb_out)
                    outs.append(out)
                return outs
            else:
                out = self.neck(x[-1])
                return out
        return x

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # print(self.search_backbone)
        if (not self.training) and self.search_backbone:
            x = self.backbone(img, self.cb_step)
        else:
            x = self.backbone(img)
        if not isinstance(x[0], (list, tuple)):
            x = [x]
        if self.with_neck:
            if self.training:
                outs = []
                for cb_out in x:
                    out = self.neck(cb_out)
                    outs.append(out)
                return outs
            else:
                out = self.neck(x[-1])
                return out
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # if not self.arch:
        #     self.set_arch({'panas_arch': [2, 2, 1, 0, 3], 'panas_d': 5, 'panas_c': 160})
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def upd_loss(self, losses, idx, weight=1): # weight是权重
        new_losses = dict()
        for k,v in losses.items():
            new_k = '{}{}'.format(k,idx) # new loss是新loss的名称，k为本来loss的名称
            # 然后编号为0的new_k是唯一的
            if weight != 1 and 'loss' in k:
                new_k = '{}_w{}'.format(new_k, weight)
            if isinstance(v,list) or isinstance(v,tuple):
                new_losses[new_k] = [i*weight for i in v]
            else:new_losses[new_k] = v*weight
        return new_losses

    # mmcv里，setarch后会调用train iter，iter会传递到forwardtrain里

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        losses = dict()
        if not isinstance(self.archs, list):
            self.archs = [self.arch]
        if self.sandwich: # 复用backbone的feature，因为只搜索neck，所以backbone不用跑4遍
            backbone_feat = self.extract_backbone_feat(img)
            # 我去掉！
            # print('backbonefeat')
        for idx, arch in enumerate(self.archs): # 遍历4个arch
            if self.search_backbone or self.search_neck or self.search_head:
                self.set_arch(arch) # 如果现在在搜索，设置网络结构
            if self.sandwich:
                xs = self.extract_neck_feat(backbone_feat) # 获得neck的特征
            else:
                xs = self.extract_feat(img)
            if len(self.archs) > 1 and self.inplace: # 蒸馏
                if idx == 0:
                    teacher_feat = xs[-1] # 采的第一个是最大子网，所以输出的feature作为teacher feature
                else:
                    kd_feat_loss = 0
                    student_feat = xs[-1]
                    for i in range(len(student_feat)): # 作为学生loss
                        kd_feat_loss += self.kd_loss(student_feat[i], teacher_feat[i].detach(), i) * self.kd_weight
                    # loss更新
                    losses.update({'kd_feat_loss_{}'.format(idx): kd_feat_loss})
                    # kd loss是每个feature依次算距离，5层距离相加算距离作为loss
                    # 可以考虑给每个stage的输出算距离
            # 后面都在更新loss
            # 这里已经实现了！FPN输出的feature做蒸馏

            # RPN的 region proposal 不用管
            for i, x in enumerate(xs):
                # RPN forward and loss
                if self.with_rpn:
                    proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                      self.test_cfg.rpn)
                    rpn_losses, proposal_list = self.rpn_head.forward_train(
                        x,
                        img_metas,
                        gt_bboxes,
                        gt_labels=None,
                        gt_bboxes_ignore=gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg,
                        **kwargs)
                    if len(xs) > 1:
                        rpn_losses = self.upd_loss(rpn_losses, idx=i + idx * len(self.archs))
                        # 给了子网的编号，表示更新该子网的loss
                    losses.update(rpn_losses)
                else:
                    proposal_list = proposals
                roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                         gt_bboxes, gt_labels,
                                                         gt_bboxes_ignore, gt_masks,
                                                         **kwargs)
                if len(xs) > 1:
                    roi_losses = self.upd_loss(roi_losses, idx=i + idx * len(self.archs))
                # 对每个子网的loss加上idx的编号，表示第n个子网的loss，
                losses.update(roi_losses)


        # for key in losses:
        #     val = losses[key]
        #     if isinstance(val, list):
        #         dtype = val[0].dtype
        #     else:
        #         dtype = val.dtype
        #     print(key, dtype)

        return losses # 传回去loss

@DETECTORS.register_module()
class CBFasterRCNN(CBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 search_backbone=False,
                 search_neck=False,
                 search_head=False,
                 sandwich=False,
                 inplace=False,
                 kd_weight=1e-8,
                 ):
        super(CBFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            search_backbone=search_backbone,
            search_neck=search_neck,
            search_head=search_head,
            sandwich=sandwich,
            inplace = inplace,
            kd_weight = kd_weight
        )

@DETECTORS.register_module()
class CBCascadeRCNN(CBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 search_backbone=False,
                 search_neck=False,
                 search_head=False,
                 sandwich=False,
                 inplace=False,
                 kd_weight=1e-8,
                 ):
        super(CBCascadeRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            search_backbone=search_backbone,
            search_neck=search_neck,
            search_head=search_head,
            sandwich=sandwich,
            inplace=inplace,
            kd_weight=kd_weight
        )

@DETECTORS.register_module()
class CBMaskRCNN(CBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 search_backbone=False,
                 search_neck=False,
                 search_head=False,
                 sandwich=False,
                 inplace=False,
                 kd_weight=1e-8,
                 ):
        super(CBMaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            search_backbone=search_backbone,
            search_neck=search_neck,
            search_head=search_head,
            sandwich=sandwich,
            inplace=inplace,
            kd_weight=kd_weight
        )

@DETECTORS.register_module()
class CBHybridTaskCascade(CBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 search_backbone=False,
                 search_neck=False,
                 search_head=False,
                 sandwich=False,
                 inplace=False,
                 kd_weight=1e-8,
                 ):
        super(CBHybridTaskCascade, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            search_backbone=search_backbone,
            search_neck=search_neck,
            search_head=search_head,
            sandwich=sandwich,
            inplace=inplace,
            kd_weight=kd_weight
        )

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
