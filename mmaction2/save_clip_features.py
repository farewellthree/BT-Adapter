import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
import mmengine
from mmaction.datasets import ActivityNetVideoDataset
from mmaction.models.recognizers import CLIPSimilarity_split

T = 100
data_root = '/Path/to/Your/ActivityNet/Raw_Frames'
ann_file = '/Path/to/Your/video_chatgpt/train_list.txt'
clip_feat_path = '/Your/Save/Path'
pretrained_path = '/Path/to/Openai-CLIP_L14/Pretrain_weights'
with open('/Path/to/Your/Video-ChatGPT/docs/train_video_ids.txt') as f:
    all_names = f.readlines()

if not os.path.exists(clip_feat_path):
    os.makedirs(clip_feat_path)
pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=T,
        test_mode=True, _scope_='mmaction'),
    dict(type='RawFrameDecode', _scope_='mmaction'),
    dict(type='Resize', scale=(-1, 256), _scope_='mmaction'),
    dict(type='CenterCrop', crop_size=224, _scope_='mmaction'),
    dict(type='FormatShape', input_format='NCHW', _scope_='mmaction'),
    dict(type='PackActionInputs', _scope_='mmaction')
]
dataset = ActivityNetVideoDataset(
        ann_file=ann_file,
        data_root=data_root,
        pipeline=pipeline,
        test_mode=True)

collate_fn = mmengine.registry.FUNCTIONS.get('pseudo_collate')
dataloader = DataLoader(batch_size=1,shuffle=False,dataset=dataset,collate_fn=collate_fn)
total = len(dataloader)

visual_encoder=dict(type='VITCLIPPretrained', clip_weight=pretrained_path)
text_encoder=dict(type='CLIPTextPretrained', clip_weight=pretrained_path)
data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.771, 116.746, 104.093],
        std=[68.500, 66.632, 70.323],
        format_shape='NCHW', _scope_='mmaction')

model = CLIPSimilarity_split(visual_encoder=visual_encoder, text_encoder=text_encoder, 
                             frozen_layers=-1, data_preprocessor=data_preprocessor, adapter=None).cuda()

model = model.half()
model.eval()

counter = 0
video_clip_features = {}
for i, data in enumerate(dataloader):
    video_id = all_names[i].strip()
    if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
        continue
    try:
        data = model.data_preprocessor(data, training=False)
        #video = video.cuda().half()
        video = data['inputs'].cuda().half()
        with torch.no_grad():
            all_patch, _ = model.backbone(video, return_all=True)
        x = all_patch[-5]
        video_clip_features[video_id] = x.detach().cpu().numpy().astype("float16")
        counter += 1
        print (counter,total)
    except Exception as e:
        print(f"Can't process {video_id}")
        print (e)
    
    if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
        for key in video_clip_features.keys():
            features = video_clip_features[key]
            with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                pickle.dump(features, f)
        video_clip_features = {}

for key in video_clip_features.keys():
    features = video_clip_features[key]
    with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
        pickle.dump(features, f)
