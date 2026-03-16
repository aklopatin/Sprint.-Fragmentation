dataset_type = 'BaseSegDataset'
data_root = 'train_dataset_for_students'

crop_size = (512, 512)
num_classes = 3
metainfo = dict(
    classes=('background', 'class_1', 'class_2'),
    palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0]])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 2048),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_root=data_root,
        data_prefix=dict(img_path='img/train', seg_map_path='labels/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_root=data_root,
        data_prefix=dict(img_path='img/val', seg_map_path='labels/val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator
