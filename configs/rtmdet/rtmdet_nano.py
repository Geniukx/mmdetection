_base_ = "./rtmdet_l_8xb32-300e_coco.py"

import os

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.25,
        use_depthwise=True,
    ),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True,
    ),
    bbox_head=dict(
        in_channels=64,
        feat_channels=64,
        share_conv=False,
        exp_on_reg=False,
        use_depthwise=True,
        num_classes=1,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

input_shape = 320

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="CachedMosaic",
        img_scale=(input_shape, input_shape),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False,
    ),
    dict(
        type="RandomResize",
        scale=(input_shape * 2, input_shape * 2),
        ratio_range=(0.5, 1.5),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(input_shape, input_shape)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="Pad", size=(input_shape, input_shape), pad_val=dict(img=(114, 114, 114))
    ),
    dict(type="PackDetInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize",
        scale=(input_shape, input_shape),
        ratio_range=(0.5, 1.5),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=(input_shape, input_shape)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="Pad", size=(input_shape, input_shape), pad_val=dict(img=(114, 114, 114))
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(input_shape, input_shape), keep_ratio=True),
    dict(
        type="Pad", size=(input_shape, input_shape), pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

data_root = '/mnt/fasthouse-00/qinglang/Pharmacy/'
anno_name = 'coco_selected.json'

metainfo = {
    'classes': (
        'hand',
    )
}

train_dataset = dict(
    _delete_=True,
    type="ConcatDataset",
    datasets=[
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file=rf'{data_name}/{anno_name}',
            data_prefix=dict(img=rf'{data_name}/images')
        )
        for data_name in os.listdir(data_root) if anno_name in os.listdir(os.path.join(data_root, data_name))
    ],
    ignore_keys=[
        'keypoints',
        'keypoints_confidence',
        'bbox_confidence',
    ],
)

test_dataset = dict(
    _delete_=True,
    type="CocoDataset",
    metainfo=metainfo,
    data_root=data_root,
    pipeline=test_pipeline,
    ann_file=rf'20240110_170139/{anno_name}',
    data_prefix=dict(img='20240110_170139/images'),
)

train_dataloader = dict(
    batch_size=64,
    dataset=train_dataset
)

val_dataloader = dict(
    batch_size=64,
    dataset=test_dataset
)

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type="PipelineSwitchHook",
        switch_epoch=4,
        switch_pipeline=train_pipeline_stage2,
    ),
]

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + rf"20240110_170139/{anno_name}",
    metric="bbox",
    format_only=False,
)

test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=0.01))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[7],
        gamma=0.1)
]

train_cfg = dict(
    max_epochs=8,
    val_interval=1,
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=-1
    ),
    logger=dict(interval=100),
)

load_from = "/mnt/nas0-pool0/personal/qinglang/checkpoints/rtmdet/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth"

# resume = True