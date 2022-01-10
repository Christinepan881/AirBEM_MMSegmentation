_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/airbem_512x512.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=3), 
    auxiliary_head=dict(num_classes=3),
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(depth=101))
