_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/scoliosis3.py',
    '../_base_/default_runtime.py'
]

checkpoint_file = '/pretrain/swin_large_patch4_window12_384_22k.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,  # base128; large192
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],  # base4, 8, 16, 32; large6, 12, 24, 48
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[192, 384, 768, 1536]),
    # base128, 256, 512, 1024; large192, 384, 768, 1536
    auxiliary_head=dict(in_channels=768))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
