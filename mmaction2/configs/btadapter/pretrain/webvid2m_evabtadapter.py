_base_ = '../../_base_/default_runtime.py'

pretrained_path = '/Path/to/EVA-CLIP_g/Pretrain_weights'
model = dict(
    type='CLIPSimilarity_split',
    visual_encoder=dict(type = 'EVAVisionTransformer_BTAdapter',
                        eva_clip = dict(type='EVAVisionTransformer', 
                        img_size=224, patch_size=14,
                        use_mean_pooling=False, embed_dim=1408,
                        depth=40, num_heads=1408//88,
                        mlp_ratio=4.3637, qkv_bias=True,
                        drop_path_rate=0.4,
                        num_classes=1024,
                        grad_checkpointing=True),
                        pretrained=pretrained_path,
                        mask_rate=0.7),
    text_encoder=dict(type='TextTransformer', width=768, heads=12, output_dim=1024),
    to_float32=True,
    frozen_layers=True,
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
    pretrain={'branch_text_nce':0.1,},
    pretrained_ckpt = pretrained_path,
    adapter=None)

val_dataset_type = 'MsrvttDataset'
val_data_root = '/Path/to/your/msrvtt/dataset'
train_data_type = 'Weivid2_5mDataset_source'
train_data_root = '/Path/to/your/webvid2m/dataset'
train_ann_file = '/Path/to/your/webvid2m/csv_annotation'
file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize', length=77),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize', length=77),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=train_data_type,
        ann_file=train_ann_file,
        data_root=train_data_root,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        ann_file='test_JSFUSION.json',
        data_root=val_data_root,
        data_prefix=dict(video='videos'),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=val_dataset_type,
        ann_file='test_JSFUSION.json',
        data_root=val_data_root,
        data_prefix=dict(video='videos'),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='RetrievalMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=1, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        by_epoch=True,
        begin=0,
        end=0.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=4.5,
        eta_min=0,
        by_epoch=True,
        begin=0.5,
        end=4,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-06,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            '.btadapter': dict(lr_mult=10.),
            'extra_proj': dict(lr_mult=10.),
    }),
    clip_grad=dict(max_norm=5, norm_type=2)
)

default_hooks = dict(checkpoint=dict(type='printBest_CheckpointHook', interval=1, save_best='auto', rule='greater'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=24*48)