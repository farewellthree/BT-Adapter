_base_ = ['../../../_base_/default_runtime.py']

pretrained_path = '/Path/to/Openai-CLIP_L14/Pretrain_weights'
# model settings
num_frames = 16
classes = '/Path/to/your/hmdb/class_name.txt'

model = dict(
    type='CLIPSimilarity_split',
    visual_encoder=dict(type='VITCLIPPretrained_BTAdapter', depth=4, clip_weight=pretrained_path),
    text_encoder=dict(type='CLIPTextPretrained', clip_weight=pretrained_path),
    class_path = classes,
    to_float32=True,
    frozen_layers=False,
    data_preprocessor=dict(
        type='MultiModalDataPreprocessor',
        preprocessors=dict(
            imgs=dict(
                type='ActionDataPreprocessor',
                mean=[122.771, 116.746, 104.093],
                std=[68.500, 66.632, 70.323],
                format_shape='NCHW'),
            text=dict(type='ActionDataPreprocessor', to_float32=False))),
    tau = 0.01,
    task = "recognition",
    adapter=None)

# dataset settings
dataset_type = 'ZeroShotClfDataset'
data_root = "/Path/to/your/hmdb/dataset/"
data_root_val = "/Path/to/your/hmdb/dataset/"
ann_file_train = '/Path/to/your/hmdb/train_tsv_annotation'
ann_file_val = '/Path/to/your/hmdb/test_tsv_annotation'
ann_file_test = '/Path/to/your/hmdb/test_tsv_annotation'
delimiter = '\t'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        delimiter=delimiter,
        label_offset=-1,
        class_path=classes,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        delimiter=delimiter,
        label_offset=-1,
        class_path=classes,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        delimiter=delimiter,
        label_offset=-1,
        class_path=classes,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='ZeroShotAccMetric')
test_evaluator = dict(type='ZeroShotAccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=55, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1e-5
optim_wrapper = dict(
    type='AmpOptimWrapper',
    #accumulative_counts=2,
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0, bias_decay_mult=0.0,
        custom_keys={
        #'STAN': dict(lr_mult=10.),
    }),
    clip_grad=dict(max_norm=20, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min_ratio=0.1,
        by_epoch=True,
        begin=5,
        end=55,
        convert_to_iter_based=True)
]

resume=False
default_hooks = dict(
    checkpoint=dict(type='printBest_CheckpointHook', interval=1, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)


