import torch
import torch.nn as nn
from typing import Any, Optional
from dataclasses import dataclass, asdict
from argparse import Namespace

import torch.nn.functional as F
from transformers import (
    WhisperForConditionalGeneration, WhisperConfig, WhisperFeatureExtractor,
    HubertModel, HubertConfig,
    AutoConfig, AutoModelForCTC, AutoModel, AutoProcessor,AutoFeatureExtractor
)

from types import SimpleNamespace

from .Qformer import BertConfig, BertLMHeadModel
     
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        # create a list of layers
        self.layers = nn.ModuleList([
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        ])
    
    def forward(self, x):
        # iterate through all the layers
        for layer in self.layers:
            x = layer(x)
        return x         
          
class Multimodal_Attention(nn.Module):
    def __init__(self, qdim, kdim, num_heads):
        super(Multimodal_Attention, self).__init__()
        # create a list of layers
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=qdim, kdim=kdim, vdim=kdim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(qdim)
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=qdim, kdim=kdim, vdim=kdim, num_heads=num_heads)
    
    def forward(self, audio_feature, visual_feature):
        # iterate through all the layers
        
        x, _ = self.mha0(query=visual_feature, key=audio_feature, value=audio_feature) # T B D
        x = x + visual_feature
        x = self.layer_norm(x)
        x2, _ = self.mha1(query=visual_feature, key=audio_feature, value=audio_feature) # T B D
        x2 = x + x2

        return x2
    
# Audio Encoder
class WhisperAudioTower(nn.Module):
    def __init__(self, audio_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)

    def load_model(self):
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(self.audio_tower_name)

        self.audio_tower = WhisperForConditionalGeneration.from_pretrained(
                self.audio_tower_name,
                dtype=torch.bfloat16,
            ).model.encoder
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, input_features, **kwargs):
        if type(input_features) is list:
            audio_features = []
            for audio in input_features:
                audio_feature = self.audio_tower(audio.to(device=self.device, dtype=self.dtype).unsqueeze(0)).last_hidden_state
                audio_features.append(audio_feature)
        else:
            audio_features = self.audio_tower(input_features.to(device=self.device, dtype=self.dtype)).last_hidden_state

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.d_model

    @property
    def num_mel_bins(self):
        return self.config.num_mel_bins
    
class HubertAudioTower(nn.Module):
    def __init__(self, audio_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
             
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = HubertConfig.from_pretrained(audio_tower)
            
    def load_model(self):
        self.audio_tower = HubertModel.from_pretrained(
                self.audio_tower_name,
                dtype=torch.bfloat16,
            )
        self.hubert_config = HubertConfig.from_pretrained(self.audio_tower_name)
        self.audio_processor = AutoFeatureExtractor.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)
        self.is_loaded = True
        
    
    @torch.no_grad
    def forward(self, input_values, padding_mask, **kwargs):
        logits = self.audio_tower(input_values=input_values.to(device=self.device, dtype=self.dtype), padding_mask=padding_mask.to(device=self.device, dtype=torch.bool))
        return logits.last_hidden_state
    
    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.hubert_config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

class ZipformerAudioTower(nn.Module):
    def __init__(self, audio_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.audio_tower_name = audio_tower

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = None  # zipformer 没标准 config 类

    def load_model(self):
        self.audio_processor = AutoFeatureExtractor.from_pretrained(
            self.audio_tower_name
        )

        self.audio_tower = AutoModel.from_pretrained(
            self.audio_tower_name,
            trust_remote_code=True
        )

        self.audio_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, input_values, padding_mask, **kwargs):
        """
        audios:
            - list[np.array] 或
            - torch.Tensor (B, T)
        """
        outputs = self.audio_tower(input_values=input_values.to(device=self.device, dtype=self.dtype),
                                   padding_mask=padding_mask.to(device=self.device, dtype=torch.bool))

        audio_features = outputs.last_hidden_state

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(
            1, 1, self.hidden_size,
            device=self.device,
            dtype=self.dtype
        )

    @property
    def dtype(self):
        return next(self.audio_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.audio_tower.parameters()).device

    @property
    def hidden_size(self):
        if hasattr(self.audio_tower.config, "feedforward_dim"):
            return self.audio_tower.config.feedforward_dim[-1]
        else:
            return 768

    @property
    def config(self):
        return self.audio_tower.config if self.is_loaded else self.cfg_only
    

def build_audio_tower(audio_tower_cfg, delay_load=False, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
            
    if "whisper" in audio_tower:
        whisper = WhisperAudioTower(audio_tower=audio_tower, delay_load=delay_load)
        whisper_cfg = whisper.config
        
        audio_tower = whisper
        audio_tower_cfg = whisper_cfg
    
    elif 'hubert' in audio_tower:
        hubert = HubertAudioTower(audio_tower=audio_tower, delay_load=delay_load)
        hubert_cfg = hubert.config
        
        audio_tower = hubert
        audio_tower_cfg = hubert_cfg
        
    elif 'zipformer' in audio_tower:
        zipformer = ZipformerAudioTower(audio_tower=audio_tower, delay_load=delay_load)
        zipformer_cfg = zipformer.config
        
        audio_tower = zipformer
        audio_tower_cfg = zipformer_cfg
    else:
        raise ValueError(f'Unknown audio tower: {audio_tower}')
    
    return audio_tower, audio_tower_cfg


def build_qformer(qformer_cfg):
    qformer_config = BertConfig.from_pretrained(qformer_cfg.qformer_model)
    qformer_config.num_hidden_layers = qformer_cfg.qformer_layers
    qformer_config.encoder_width = qformer_cfg.embed
    qformer_config.hidden_size = qformer_cfg.qformer_dim 
    qformer_config.add_cross_attention = True
    qformer_config.cross_attention_freq = 1
    qformer_config.query_length = qformer_cfg.max_queries
    qformer = BertLMHeadModel(qformer_config)
        
    return qformer, qformer_config

if __name__ == "__main__":
    audio_tower, audio_tower_cfg = build_audio_tower(SimpleNamespace(audio_tower="reazon-research/japanese-zipformer-base-k2-rs35kh-bpe"))
    print(audio_tower)
    print(audio_tower_cfg)
    x = torch.randn(1, 80, 3000)
    out = audio_tower(x)
    print(out.shape)