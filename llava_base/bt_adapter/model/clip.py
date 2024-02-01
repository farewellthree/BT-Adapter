from webbrowser import get
import torch
import torch.nn as nn
from torch.utils import checkpoint
import numpy as np

from einops import rearrange

from transformers import CLIPModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP
from transformers.modeling_outputs import BaseModelOutputWithPooling
from mmcv.cnn.bricks import DropPath
from mmengine.model import constant_init
from mmengine.runner.checkpoint import _load_checkpoint

class CLIPPretrained(nn.Module):
    def __init__(self, clip_model, depth=4, share_spatial=False, clip_weight='/group/30042/ruyangliu/mmaction2/ckpt/clip/L14', 
                 reserve_layers=-1, return_layers=0, clip_cls=False, gradient_checkpointing=False, **kwargs): 

        super().__init__()
        
        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
        else:
            configuration = CLIPConfig().from_pretrained('openai/clip-vit-large-patch14')

        self.depth = depth
        dpr = np.linspace(0, 0.1, depth)

        self.reserve_layers = reserve_layers
        self.return_layers = return_layers
        self.clip_cls = clip_cls
        self.num_patches = (configuration.vision_config.image_size // configuration.vision_config.patch_size) ** 2
        self.embed_dim = configuration.vision_config.hidden_size
        self.share_spatial = share_spatial
        self.gradient_checkpointing = gradient_checkpointing

        #self.visual_projection = clip_model.visual_projection
        if self.reserve_layers > 0:
            assert self.reserve_layers>=self.depth
            self.layers = clip_model.vision_model.encoder.layers[-self.reserve_layers:return_layers+1]
            self.depth = self.depth + return_layers + 1
        else:
            self.layers = clip_model.vision_model.encoder.layers
            self.class_embedding = clip_model.vision_model.embeddings.class_embedding
            self.patch_embedding = clip_model.vision_model.embeddings.patch_embedding
            self.pre_layrnorm = clip_model.vision_model.pre_layrnorm
            self.post_layernorm = clip_model.vision_model.post_layernorm
            self.position_embedding = clip_model.vision_model.embeddings.position_embedding

    def forward_embedding(self, x):
        batch_size = x.shape[0]
        patch_embeds = self.patch_embedding(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_ids = torch.arange(self.num_patches + 1, dtype=torch.long, device=x.device).expand((1, -1))
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

    def forward_patch(self, x, attention_mask=None):
        if getattr(self, 'pre_layrnorm', None) is not None:
            x = self.pre_layrnorm(x)
        
        encoder_states = ()
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (x,)

            layer_outputs = encoder_layer(x, attention_mask = attention_mask, causal_attention_mask=None)
            x = layer_outputs[0]

        if self.reserve_layers>0:
            encoder_states = encoder_states + (x,)
            return encoder_states, None
        encoder_states = encoder_states + (x,)
        #encoder_states = encoder_states + (x+x2,)
        cls_token = x[:, 0] 
        cls_token = self.post_layernorm(cls_token)

        return encoder_states, cls_token

    def forward(self, x, return_all=False, **kwargs):
        if x.ndim == 5:
            # B, 3, num_frames, 224, 224
            if x.shape[1]==3:
                B, D, T, H, W = x.shape             
                x = x.permute(0, 2, 1, 3, 4)
            else:
                B, T, D, H, W = x.shape   
            x = x.reshape((-1,) + x.shape[2:])
        elif x.ndim == 4:
            if x.shape[1]==3:
                T, _, _, _ = x.shape
            else:
                _, T, _, _ = x.shape
                x = x.reshape((-1,) + x.shape[2:])
        elif x.ndim == 3:
            T, _, _ = x.shape
        self.T = T
        
        # vision masks ensemble into clip model
        if self.reserve_layers < 0:
            x = self.forward_embedding(x)
        vision_outputs = self.forward_patch(x)
        #x_tokens, x_cls = self.result_process(vision_outputs)
        
        x_tokens, x_cls = vision_outputs

        return BaseModelOutputWithPooling(
            pooler_output=x_cls,
            hidden_states=x_tokens,
        )

def build_CLIP(clip_model, weight=None, training=False):  
    if training:
        model = CLIPPretrained(clip_model, reserve_layers=4, return_layers=-2, share_spatial=False, 
                                       clip_cls=True, gradient_checkpointing=True)
    else:
        model = CLIPPretrained(clip_model, share_spatial=False, clip_cls=False)
    if weight is not None:
        if weight[-3:] == 'pth':
            state_dict = _load_checkpoint(weight)['state_dict']
        else:
            state_dict = torch.load(weight, map_location='cpu')
        if 'model.vision_tower.' in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k,v in state_dict.items():
                new_state_dict[k.replace('model.vision_tower.','')] = v
            state_dict = new_state_dict 

        status = model.load_state_dict(state_dict, strict=False)
        if status.unexpected_keys:
            print ('------------------------------------------------')
            print(f"Unexpected Keys: {status.unexpected_keys}.")
        if status.missing_keys:
            print ('------------------------------------------------')
            print(f"Missing_keys Keys: {status.missing_keys}.")
    return model

