_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/scoliosis3.py',
    '../_base_/default_runtime.py'
]

model = dict(test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
