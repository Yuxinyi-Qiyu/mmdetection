custom_imports=dict(imports='mmdet_custom.datasets', allow_failed_imports=False)

# dataset settings
img_scale = (320, 256)

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
fake_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'SHIPDataset8'
data_root = 'data/ship-comp-committee-new/'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='RepeatDataset',
        times=1,  # simply change this from 1 to 16 for 100e - 1600e training.
        dataset=dict(
            pipeline=train_pipeline,
            type=dataset_type,
            ann_file=data_root + 'train.txt',
            img_prefix=data_root)),
    # train=dict(
    #     pipeline=fake_train_pipeline,
    #     type=dataset_type,
    #     ann_file=data_root + 'val.txt',
    #     img_prefix=data_root),
    train_val=dict(
        type=dataset_type,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    val=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root),
    test=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root))