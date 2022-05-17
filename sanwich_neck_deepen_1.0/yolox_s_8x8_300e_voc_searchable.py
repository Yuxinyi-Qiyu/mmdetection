optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(
    type='EpochBasedRunnerSuper',
    max_epochs=300,
    widen_factor_range=[0.125, 0.25, 0.375, 0.5],
    deepen_factor_range=[0.33, 0.67, 1.0],
    search_backbone=True,
    search_neck=True,
    search_head=True,
    sandwich=True)
checkpoint_config = dict(interval=50)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
widen_factor_range = [0.125, 0.25, 0.375, 0.5]
widen_factor_backbone = [0.5, 0.5, 0.5, 0.5, 0.5]
deepen_factor_range = [0.33, 0.67, 1.0]
deepen_factor = [1.0, 1.0, 1.0, 1.0]
widen_factor_neck = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
widen_factor_neck_out = 0.5
search_backbone = True
search_neck = True
search_head = True
sandwich = True
inplace = 'NonLocal'
img_scale = (640, 640)
find_unused_parameters = True
model = dict(
    type='YOLOX_Searchable_Sandwich',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    inplace='NonLocal',
    search_backbone=True,
    search_neck=True,
    search_head=True,
    backbone=dict(
        type='CSPDarknet_Searchable',
        conv_cfg=dict(type='USConv2d'),
        norm_cfg=dict(type='USBN2d'),
        deepen_factor=[1.0, 1.0, 1.0, 1.0],
        widen_factor=[0.5, 0.5, 0.5, 0.5, 0.5]),
    neck=dict(
        type='YOLOXPAFPN_Searchable',
        conv_cfg=dict(type='USConv2d'),
        norm_cfg=dict(type='USBN2d'),
        in_channels=[128, 256, 512],
        out_channels=128,
        widen_factor=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        widen_factor_out=0.5,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead_Searchable',
        num_classes=20,
        in_channels=128,
        widen_factor_neck=0.5,
        feat_channels=128,
        conv_cfg=dict(type='USConv2d'),
        norm_cfg=dict(type='USBN2d')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/VOCdevkit/'
dataset_type = 'VOCDataset'
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='VOCDataset',
        ann_file=[
            'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
            'data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
        ],
        img_prefix=['data/VOCdevkit/VOC2007/', 'data/VOCdevkit/VOC2012/'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='VOCDataset',
            ann_file=[
                'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
                'data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=['data/VOCdevkit/VOC2007/', 'data/VOCdevkit/VOC2012/'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    train_val=dict(
        type='VOCDataset',
        ann_file=[
            'data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
            'data/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
        ],
        img_prefix=['data/VOCdevkit/VOC2007/', 'data/VOCdevkit/VOC2012/'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    val=dict(
        type='VOCDataset',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='VOCDataset',
        ann_file='data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
max_epochs = 300
num_last_epochs = 15
interval = 10
evaluation = dict(save_best='auto', interval=10, metric='mAP')
work_dir = 'sanwich_neck_deepen_1.0'
auto_resume = False
gpu_ids = range(0, 8)
