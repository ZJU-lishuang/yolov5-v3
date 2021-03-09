_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(
    head=dict(
        num_classes=5,
        topk=(1, 2),
    ))

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        data_prefix='/home/lishuang/Disk/shengshi_data/new_anti_tail/classificationtotal/train',),
    val=dict(
        data_prefix='/home/lishuang/Disk/shengshi_data/new_anti_tail/classificationtotal/validation',
        ann_file=None,),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',))
evaluation = dict(interval=1, metric='accuracy')

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

