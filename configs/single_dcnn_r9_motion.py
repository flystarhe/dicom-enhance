# model settings
model = dict(
    type='DCNN',
    net=dict(
        type='ResNet',
        depth=9,
        in_channels=1,
        out_channels=256,
        padding_type='reflect',
        upsampling='bilinear'),
)
# model training and testing settings
train_cfg = dict()
test_cfg = dict()
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
train_pipeline = [
    dict(type='LoadDicomFromFile'),
]
test_pipeline = [
    dict(type='LoadDicomFromFile'),
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='coco_train.json',
        data_root=data_root,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='coco_test.json',
        data_root=data_root,
        pipeline=test_pipeline)
)
# optimizer

# learning policy

# runtime settings
