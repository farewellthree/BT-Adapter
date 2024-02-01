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

class VITCLIPPretrained_BTAdapter(nn.Module):
    def __init__(self, clip_model, depth=4, clip_weight='/group/30042/ruyangliu/mmaction2/ckpt/clip/L14', 
                 reserve_layers=-1, return_layers=0, gradient_checkpointing=False,
                  out_norm=None, middle_norm=None, **kwargs): 

        super().__init__()
        
        if clip_weight:
            configuration = CLIPConfig().from_pretrained(clip_weight)
        else:
            configuration = CLIPConfig().from_pretrained('openai/clip-vit-large-patch14')

        self.depth = depth
        dpr = np.linspace(0, 0.1, depth)

        self.reserve_layers = reserve_layers
        self.return_layers = return_layers
        self.num_patches = (configuration.vision_config.image_size // configuration.vision_config.patch_size) ** 2
        self.embed_dim = configuration.vision_config.hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        self.position_embedding = clip_model.vision_model.embeddings.position_embedding
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

        self.out_norm = out_norm
        self.middle_norm = middle_norm
        if out_norm is not None:
            self.balance1 = nn.Parameter(torch.zeros((self.embed_dim)))
            self.sigmoid = nn.Sigmoid()
        if middle_norm is not None:
            self.balance2 = nn.Parameter(torch.zeros((self.embed_dim)))
            self.sigmoid = nn.Sigmoid()
        self.btadapter_S_layers = nn.ModuleList([CLIPLayer_Spatial(configuration.vision_config, 8, dpr[i]) for i in range(self.depth)])
        self.btadapter_T_layers = nn.ModuleList([CLIPLayer_AttnTime(configuration.vision_config, 8, dpr[i]) for i in range(self.depth)])
        
        #self.btadapter_pos_embed = nn.Embedding(self.num_patches + 1, self.embed_dim)
        self.btadapter_time_embed = nn.Embedding(64, self.embed_dim)
        
        self.btadapter_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #self.btadapter_cls_token = nn.Parameter(torch.zeros(self.embed_dim))
        self.drop_after_pos = nn.Dropout(p=0)
        self.drop_after_time = nn.Dropout(p=0)

        self.init_weights()

    def init_weights(self):
        total_depth = len(self.layers)
        layer_para = self.layers.state_dict()
        spatial_para = {}
        load_start = total_depth - self.depth
        for k, v in layer_para.items():
            num_layer = int(k.split(".")[0])
            if num_layer >= load_start:
                spatial_para[k.replace(str(num_layer),str(num_layer-load_start),1)] = v.clone()
        self.btadapter_S_layers.load_state_dict(spatial_para)

    def forward_embedding(self, x):
        batch_size = x.shape[0]
        patch_embeds = self.patch_embedding(x)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_ids = torch.arange(self.num_patches + 1, dtype=torch.long, device=x.device).expand((1, -1))
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings
    
    def get_combined_embedding(self,x,x2):
        x_patch, x2_patch = x[:,1:,], x2[:,1:,]

        x_cls = rearrange(x[:,:1], '(b t) p m -> b t p m', t=self.T).mean(dim=1)
        x2_cls = x2[:,:1,]
        x_patch = rearrange(x_patch, '(b t) p m -> b t p m', t=self.T).mean(dim=1)
        p = x_patch.size(1)
        x2_patch = rearrange(x2_patch, 'b (p t) m -> b t p m', p=p).mean(dim=1)
        
        combine = torch.cat(((x_patch+x2_patch)/2,(x_cls+x2_cls)/2),dim=1)

        #x_patch = rearrange(x_patch, '(b t) p m -> b t p m', t=self.T).mean(dim=1)
        #x2_patch = rearrange(x2_patch, 'b (p t) m -> b t p m', t=self.T).mean(dim=1)
        #combine = (x_patch+x2_patch)/2
        return combine

    def forward_patch(self, x, attention_mask=None):
        if getattr(self, 'pre_layrnorm', None) is not None:
            x = self.pre_layrnorm(x)
        total = len(self.layers)
        x2 = None
        encoder_states = ()
        for idx, encoder_layer in enumerate(self.layers):
            if x2 is None:
                encoder_states = encoder_states + (x,)
            else:
                combine = self.get_combined_embedding(x,x2)
                encoder_states = encoder_states + (combine,)

            #if self.return_layers != 0:
            #    if idx > total + self.return_layers:
            #        return encoder_states, None
            layer_outputs = encoder_layer(x, attention_mask = attention_mask, causal_attention_mask=None)
            x = layer_outputs[0]

            if idx >= total-self.depth:
                num_layer = idx + self.depth - total
                x2 = self.forward_BTAdapter(x, x2, num_layer, self.T)
        
        if self.reserve_layers>0:
            combine = self.get_combined_embedding(x,x2)
            encoder_states = encoder_states + (combine,)
            return encoder_states, None
        encoder_states = encoder_states + ((x,x2),)
        #encoder_states = encoder_states + (x+x2,)
        if self.out_norm is not None:
            weight = self.sigmoid(self.out_norm * self.balance1)
            cls_token = ((1 - weight) * x2[:, 0].repeat(1, self.T).view(x2.size(0) * self.T, -1) + weight * x[:, 0])
        else:
            cls_token = x[:, 0] + x2[:, 0].repeat(1, self.T).view(x2.size(0) * self.T, -1)
        cls_token = self.post_layernorm(cls_token)

        return encoder_states, cls_token

    def forward_BTAdapter(self, x1, x2, num_layer, T):
        x1 = rearrange(x1, '(b t) l d -> b t l d', t=T)
        if T>self.btadapter_time_embed.num_embeddings:
            x1 = rearrange(x1, 'b (t1 t2) l d -> b t1 t2 l d', t1=T//2, t2=2).mean(dim=2)
        T = T//2
        self.btadapter_S_layers[num_layer].t = T
        self.btadapter_T_layers[num_layer].t = T

        if x2 is not None:
            cls_token_ori = x1[:, :, 0, :]
            cls_token = cls_token_ori.mean(dim=1).unsqueeze(1)
            x1 = x1[:, :, 1:, :]
            x1 = rearrange(x1, 'b t l d -> b (l t) d')
            x1 = torch.cat((cls_token, x1), dim=1)
            
            if self.middle_norm is not None:
                weight = self.sigmoid(self.middle_norm * self.balance2)
                x = ((1 - weight) * x2 + weight * x1) 
            else:
                x = x2 + x1
            
        else:
            x = x1
        
        if num_layer==0:
            x = self.input_ini(x)

        if self.gradient_checkpointing and self.training:  
            x = checkpoint.checkpoint(self.btadapter_T_layers[num_layer],x)
            x = checkpoint.checkpoint(self.btadapter_S_layers[num_layer],x,None,None)
        else:
            x = self.btadapter_T_layers[num_layer](x)
            x = self.btadapter_S_layers[num_layer](x, None, None)
        return x 

    def input_ini(self, x):
        cls_old = x[:, :, 0, :].mean(dim=1).unsqueeze(1)
        x = x[:,:,1:,:]
        B,T,L,D = x.size()
        x = rearrange(x, 'b t l d -> (b t) l d')
        #cls_tokens = self.class_embedding.expand(x.size(0), 1, -1)
        cls_tokens = self.btadapter_cls_token.expand(x.size(0), 1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.position_embedding(position_ids)
        x = x + pos_embed
        x = self.drop_after_pos(x)
        cls = x[:B, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) l d -> (b l) t d', b=B)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        time_embed = self.btadapter_time_embed(position_ids)
        x = x + time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b l) t d -> b (l t) d', b=B)
        cls = (cls_old + cls) / 2
        x = torch.cat((cls, x), dim=1)
        return x 

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
    
class CLIPLayer_Spatial(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1, num_cls=1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = DropPath(
            layer_num) if layer_num > 0. else nn.Identity()
        self.t = T

        self.num_cls = num_cls
       
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ):
        residual = hidden_states

        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_s = hidden_states[:, 1:, :]
        
        init_cls_token = hidden_states[:, :self.num_cls, :]
        query_s = hidden_states[:, self.num_cls:, :]

        b, pt, m = query_s.size()
        p, t = pt // self.t, self.t
        cls_token = init_cls_token.unsqueeze(1).repeat(1, t, 1, 1).reshape(b * t, self.num_cls, m) #can I do?
        #cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        hidden_states = torch.cat((cls_token, query_s), 1)
        #hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
        )

        res_spatial = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        cls_token = res_spatial[:, :self.num_cls, :].reshape(b, self.t, self.num_cls, m)
        cls_token = torch.mean(cls_token, 1)
        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, self.num_cls:, :], '(b t) p m -> b (p t) m', p=p, t=self.t)
        hidden_states = torch.cat((cls_token, res_spatial), 1)
        #hidden_states = self.process.after(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CLIPLayer_AttnTime(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1, num_cls=1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.num_cls = num_cls

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = DropPath(
            layer_num) if layer_num > 0. else nn.Identity()
        #-1.0633e-04,  2.3007e-04, -6.0737e-05,  ...,  2.2769e-05,
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        constant_init(self.temporal_fc, val=0, bias=0)
        self.t = T 

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        causal_attention_mask=None,
    ):
        residual = hidden_states[:, self.num_cls:, :]


        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_t = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, :self.num_cls, :]
        query_t = hidden_states[:, self.num_cls:, :]
        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.t, self.t
        hidden_states = query_t.reshape(b * p, t, m)

        #init_cls_token, hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
        )

        res_temporal = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)
        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        hidden_states = res_temporal.reshape(b, p * self.t, m)
        #hidden_states = self.process.after(hidden_states)


        hidden_states = residual + hidden_states
        hidden_states = torch.cat((init_cls_token, hidden_states), 1)
        outputs = hidden_states

        return outputs

def build_btadapter_from_CLIP(clip_model, weight=None, training=False, use_norm=True):  
    if use_norm:
        out_norm, middle_norm = 1e4, 1e2
    else:
        out_norm, middle_norm = None, None
    if training:
        model = VITCLIPPretrained_BTAdapter(clip_model, reserve_layers=4, return_layers=-2,# gradient_checkpointing=True)
                                            out_norm=out_norm, middle_norm=middle_norm, gradient_checkpointing=True)
    else:
        model = VITCLIPPretrained_BTAdapter(clip_model, out_norm=out_norm, middle_norm=middle_norm,)
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
        if 'backbone.' in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k,v in state_dict.items():
                k = k.replace('backbone.','')
                if 'timesFPN' in k:
                    k = k.replace('timesFPN','btadapter')
                new_state_dict[k] = v
            state_dict = new_state_dict 

        status = model.load_state_dict(state_dict, strict=False)
        if status.unexpected_keys:
            print ('------------------------------------------------')
            print(f"Unexpected Keys: {status.unexpected_keys}.")
        if status.missing_keys:
            print ('------------------------------------------------')
            print(f"Missing_keys Keys: {status.missing_keys}.")
    return model

