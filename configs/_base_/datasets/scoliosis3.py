# dataset settings
dataset_type = 'ScoliosisDataset3'
data_root = '/data/scoliosis3classes'
img_norm_cfg = dict(  # 图像归一化配置
    mean=[75.99, 75.99, 75.99], std=[84.15, 84.15, 84.15], to_rgb=False)  # std：标准差
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),  # 注意color_type变量
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),  # 数据增广的比例范围
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪
    dict(type='RandomFlip', flip_ratio=0.5),  # 随机翻转
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(
        type='MultiScaleFlipAug',  # 封装测试时的数据增广
        img_scale=(2048, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # img_ratios=[1.0],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True, size_divisor=32),
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,  # mini-batch
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/train',
        ann_dir='groundtruth/train',
        split='train.txt',  # split 拼接数据集
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/test',
        ann_dir='groundtruth/test',
        split='test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/test',
        ann_dir='groundtruth/test',
        split='test.txt',
        pipeline=test_pipeline))
