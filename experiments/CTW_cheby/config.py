# model settings
train_size=960
cheby_degree = 44
model = dict(
    type='ChebyRPN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        type='ChebyRPNHead',
        du_cfg=dict(in_out_channels=256, kernel_size=(1, 9), groups=256),
        lr_cfg=dict(in_out_channels=256, kernel_size=(9, 1), groups=256),
        in_channels=256,
        feat_channels=256,
        anchor_scales=[1],
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64],
        num_coords=cheby_degree+4, 
        target_means=0.0,
        target_stds=1.0,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='ContentLoss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_ctr=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# model training and testing settings
# cudnn_benchmark = True
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='CenterAssigner', #MaxIoUAssigner
            level_assign=True,    # multi-level training for each branch
            centerness_assign=True, # centerness assign, iou of positives in [0, 1]
            pos_iou_thr=0.4,      # 0.5 for MaxIoUAssigner
            neg_iou_thr=0.1,
            min_pos_iou=.0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='CurveWeightedSampler', # random sapmling weighted by centerness/iou
            num=256,   #64
            pos_fraction=0.5,
            neg_pos_ub= 3,
            add_gt_as_proposals=False),
        allowed_border=0,
        use_centerness=True,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=True,
        nms_pre=1000,
        nms_post=500,
        max_num=500,
        nms_thr=0.7,
        min_bbox_size=0))
# dataset settings
dataset_type = 'CTW1500'
data_root = '../../data/CTW1500/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadCurveAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=0),
    dict(type='CurveRandomCrop', final_size=train_size, scale_range=(0.5, 5)),
    dict(type='CurveResize', img_scale=(train_size, train_size), keep_ratio=True),
    dict(type='CurveRandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CurvePad', size_divisor=32),
    dict(type='DefaultCurveFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_coefs', 'gt_skeleton']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'ImageSets/Main/train.txt',
            img_prefix=data_root,
            cache_root = 'Cache',
            encoding='cheby',
            degree=cheby_degree,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        img_prefix=data_root,
        cache_root = 'Cache',
        test_mode = True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)  # lr=0.08 if without pretrain
# runner configs
optimizer_config = dict(grad_clip=dict(max_norm=5., norm_type=2))
lr_config = dict(
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1.0 / 10,
    policy='cosine',
)
#     policy='step',
#     step=[100, 150, 175]) #[150, 225, 275])
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 500
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs'
load_from = '../Pretrain_CTW/work_dirs/latest.pth' 
resume_from = None #'./work_dirs/latest.pth'
workflow = [('train', 1)]