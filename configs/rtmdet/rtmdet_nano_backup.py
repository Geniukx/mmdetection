_base_ = "./rtmdet_l_8xb32-300e_coco.py"


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

data_root = "/mnt/nas0-pool0/datasets/Pharmacy/"

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
            ann_file='20240110_164259/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_164259/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_151842/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_151842/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_153727/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_153727/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_160543/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_160543/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00004/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00004/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='IMG_6903/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='IMG_6903/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_145107/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_145107/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00010/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00010/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00009/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00009/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00003/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00003/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_145726/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_145726/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_161038/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_161038/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00002/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00002/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_153022/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_153022/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_165145/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_165145/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_165438/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_165438/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_162030/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_162030/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_164732/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_164732/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_151727/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_151727/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00007/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00007/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00011/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00011/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_151355/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_151355/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_155131/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_155131/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_155855/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_155855/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00012/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00012/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='IMG_6908/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='IMG_6908/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_155016/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_155016/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='IMG_6904/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='IMG_6904/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00001/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00001/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='VID00008/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='VID00008/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_160949/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_160949/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_170021/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_170021/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_165113/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_165113/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_153843/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_153843/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_155038/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_155038/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_161712/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_161712/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_165757/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_165757/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='IMG_6902/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='IMG_6902/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_145841/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_145841/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_145927/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_145927/images')
        ),
        dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='20240110_162340/coco_bbox_0.5_filtered.json',
            data_prefix=dict(img='20240110_162340/images')
        ),
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
    ann_file='20240110_170139/coco_bbox_0.5_filtered.json',
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
    ann_file=data_root + "20240110_170139/coco_bbox_0.5_filtered.json",
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
        max_keep_ckpts=3
    ),
    logger=dict(interval=100),
)

load_from = "/mnt/nas0-pool0/personal/qinglang/checkpoints/rtmdet/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth"

# resume = True