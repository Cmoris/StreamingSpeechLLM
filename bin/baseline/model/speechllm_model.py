import os
import math
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

from .submodels.projector import build_audio_projector
from .submodels.modules import build_audio_tower, build_qformer, Projector

import sys
sys.path.append("../")
from constants import IGNORE_INDEX, MODAL_INDEX_MAP

class AudioMetaModel:
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "mm_audio_tower"):
            self.audio_tower, audio_tower_cfg = build_audio_tower(config, delay_load=True)
            self.mm_projector_a = build_audio_projector(config)
            
        if hasattr(config, "mm_qformer_model") and getattr(config, "use_qformer", True):
            self.Qformer, self.qformer_config = build_qformer(config)
            self.mm_projector = Projector(input_dim=config.qformer_dim,
                                        hidden_dim=math.floor((config.qformer_dim + config.hidden_size) / 2),
                                        output_dim=config.hidden_size)
            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.max_queries, config.qformer_hidden_size, requires_grad=True)
            )
            self.ln_head = nn.LayerNorm(config.embed)
            
        else:
            if hasattr(config, "embed"): 
                self.mm_projector = Projector(input_dim=config.embed,
                                            hidden_dim=math.floor((config.embed + config.hidden_size) / 2),
                                            output_dim=config.hidden_size)

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if isinstance(audio_tower, list):
            audio_tower = audio_tower[0]
        return audio_tower
    
    def get_Qformer(self):
        qformer = getattr(self, "Qformer", None)
        if isinstance(qformer, list):
            qformer = qformer[0]    
        return qformer
    
    def initialize_qformer(self, model_args, fsdp=None):
        pretrain_qformer = model_args.pretrain_qformer
        self.config.max_queries = int(model_args.queries_per_sec * model_args.audio_frame_rate)
        self.config.use_qformer = model_args.use_qformer
        self.config.queries_per_sec = model_args.queries_per_sec
        self.config.qformer_layers = model_args.qformer_layers
        self.config.qformer_dim = model_args.qformer_dim 
        self.config.qformer_model = model_args.qformer_model
        self.config.embed = self.config.mm_hidden_size_a
        self.config.mm_qformer_model = model_args.qformer_model
        self.config.window_level_Qformer = model_args.window_level_Qformer

        if self.config.window_level_Qformer:
            self.config.second_per_window = model_args.second_stride
            self.config.second_stride = model_args.second_stride

        if self.get_Qformer() is None:
            self.Qformer, qformer_config = build_qformer(self.config)
        else:
            qformer_config = self.qformer_config
        
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None
            
        self.config.qformer_hidden_size = qformer_config.hidden_size
        
        if getattr(self, "query_tokens", None) is None:
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.config.max_queries, qformer_config.hidden_size)
            )
        
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
        
        self.ln_head = nn.LayerNorm(self.config.embed)
        
        if pretrain_qformer is not None:
            weights = torch.load(pretrain_qformer, map_location='cpu')

            def get_w(w, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in w.items() if keyword in k}

            self.Qformer.load_state_dict(get_w(weights, 'Qformer'), strict=True)
            self.query_tokens.data.copy_(weights['query_tokens'])

    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.audio_frame_rate = model_args.audio_frame_rate
        self.config.mm_audio_tower = audio_tower

        if self.get_audio_tower() is None:
            audio_tower, audio_tower_cfg = build_audio_tower(model_args)
            self.audio_tower = [audio_tower] if fsdp else audio_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_tower = self.audio_tower[0]
                audio_tower_cfg = self.audio_tower_cfg[0]
            else:
                audio_tower = self.audio_tower
                audio_tower_cfg = self.audio_tower_cfg
                
            audio_tower.load_model()

        self.config.mm_projector_a_type = getattr(model_args, 'mm_projector_a_type', 'linear')
        self.config.mm_hidden_size_a = audio_tower.hidden_size

        if getattr(self, 'mm_projector_a', None) is None:
            self.mm_projector_a = build_audio_projector(self.config)
        else:
            for p in self.mm_projector_a.parameters():
                p.requires_grad = True
                
        if pretrain_mm_mlp_adapter is not None:
            weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(w, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in w.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(weights, 'mm_projector'), strict=True)

    def initialize_projector(self, model_args, fsdp=None):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        
        if not getattr(model_args, "use_qformer"):
            self.config.use_qformer = model_args.use_qformer
            self.config.queries_per_sec = model_args.queries_per_sec
                
        if getattr(self, "mm_projector", None) is None and getattr(model_args, "use_qformer"):
            self.mm_projector = Projector(input_dim=self.config.qformer_dim ,
                                            hidden_dim=math.floor((self.config.qformer_dim + self.config.hidden_size)/2),
                                            output_dim=self.config.hidden_size)
        elif getattr(self, "mm_projector", None) is None:
            self.mm_projector = Projector(input_dim=self.config.embed,
                                        hidden_dim=math.floor((self.config.embed + self.config.hidden_size)/2),
                                        output_dim=self.config.hidden_size)
            
        if pretrain_mm_mlp_adapter is not None:
            weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(w, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in w.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(weights, 'mm_projector'), strict=True)


class AudioMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()
    
    def get_qformer(self):
        return self.get_model().get_Qformer()
    
    def encode_audios(self, audios: dict):
        audio_tower = self.get_audio_tower()

        if audios is None:
            raise ValueError("Audio must be provided.")

        with torch.no_grad():
            audio_enc_out = audio_tower(**audios)  # (B, T, D)
        
        # ============================
        # 1. Feature length
        # ============================
        feat_lengths = [audio_enc_out.size(1)] * audio_enc_out.size(0)
        # ============================
        # 2. Audio feature projection
        # ============================
        audio_enc_out = self.get_model().mm_projector_a(
            audio_enc_out.transpose(1, 2)
        ).transpose(1, 2)  # (B, T, D') 
        # ============================
        # 3. No video → audio only
        # ============================
        av_feat = audio_enc_out
        # ============================
        # 4. Qformer / direct projection
        # ============================
        if self.config.use_qformer:
            query_output = self.compression_using_qformer(
                av_feat
            )
            query_output = self.get_model().mm_projector(query_output)
        else:
            query_output = self.get_model().mm_projector(av_feat)

        return query_output
    
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        audios: dict,
    ):
        mm_features = self.encode_audios(audios)

        mm_token_ids = list(MODAL_INDEX_MAP.values())

        def find_mm_token_indices(cur_input_ids):
            masks = [(cur_input_ids == mm_token_idx) for mm_token_idx in mm_token_ids]
            if len(masks) == 1:
                return torch.where(masks[0])[0]
            return torch.where(torch.stack(masks, dim=0).any(dim=0))[0]

        total_mm_tokens = 0
        per_sample_mm_indices = []

        for cur_input_ids in input_ids:
            mm_token_indices = find_mm_token_indices(cur_input_ids)
            per_sample_mm_indices.append(mm_token_indices)
            total_mm_tokens += mm_token_indices.numel()

        assert total_mm_tokens == mm_features.shape[0], (
            f"Mismatch between multimodal tokens and audio features: "
            f"total_mm_tokens={total_mm_tokens}, "
            f"mm_features.shape[0]={mm_features.shape[0]}. "
            f"input_ids.shape={tuple(input_ids.shape)}"
        )

        new_input_embeds = []
        new_labels = [] if labels is not None else None

        cur_mm_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_embeds = self.get_model().embed_tokens(
                cur_input_ids.clamp(min=0, max=self.vocab_size - 1)
            )

            cur_new_input_embeds = []

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []

            mm_token_indices = find_mm_token_indices(cur_input_ids)

            while mm_token_indices.numel() > 0:
                mm_token_start = mm_token_indices[0]

                cur_mm_features = mm_features[cur_mm_idx]

                # text before audio
                cur_new_input_embeds.append(cur_input_embeds[:mm_token_start])

                # audio features
                # cur_mm_features: [T_audio, hidden]
                if cur_mm_features.dim() == 3 and cur_mm_features.size(0) == 1:
                    cur_mm_features = cur_mm_features.squeeze(0)

                cur_new_input_embeds.append(cur_mm_features)

                if labels is not None:
                    # labels before audio token
                    cur_new_labels.append(cur_labels[:mm_token_start])

                    # audio feature positions 不参与 loss
                    cur_new_labels.append(
                        torch.full(
                            (cur_mm_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )

                    # skip the multimodal token itself
                    cur_labels = cur_labels[mm_token_start + 1:]

                # skip the multimodal token itself
                cur_input_ids = cur_input_ids[mm_token_start + 1:]
                cur_input_embeds = cur_input_embeds[mm_token_start + 1:]

                mm_token_indices = find_mm_token_indices(cur_input_ids)

                cur_mm_idx += 1

            # remaining text after last audio token
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(cur_input_embeds)

                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds if x.numel() > 0
            ]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

            if labels is not None:
                cur_new_labels = [x for x in cur_new_labels if x.numel() > 0]
                new_labels.append(torch.cat(cur_new_labels, dim=0))

        assert cur_mm_idx == mm_features.shape[0], (
            f"Not all audio features were consumed: "
            f"cur_mm_idx={cur_mm_idx}, mm_features={mm_features.shape[0]}"
        )

        max_len = max(x.shape[0] for x in new_input_embeds)

        padded_input_embeds = []
        padded_labels = [] if labels is not None else None
        padded_attention_mask = [] if attention_mask is not None else None

        for batch_idx, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            pad_len = max_len - cur_len

            if pad_len > 0:
                embed_pad = torch.zeros(
                    (pad_len, cur_new_embed.shape[1]),
                    dtype=cur_new_embed.dtype,
                    device=cur_new_embed.device,
                )
                cur_new_embed = torch.cat([cur_new_embed, embed_pad], dim=0)

            padded_input_embeds.append(cur_new_embed)

            if labels is not None:
                cur_new_label = new_labels[batch_idx]
                label_pad_len = max_len - cur_new_label.shape[0]

                if label_pad_len > 0:
                    label_pad = torch.full(
                        (label_pad_len,),
                        IGNORE_INDEX,
                        dtype=cur_new_label.dtype,
                        device=cur_new_label.device,
                    )
                    cur_new_label = torch.cat([cur_new_label, label_pad], dim=0)

                padded_labels.append(cur_new_label)

            if attention_mask is not None:
                cur_attention_mask = torch.cat(
                    [
                        torch.ones(
                            cur_len,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        torch.zeros(
                            max_len - cur_len,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=0,
                )
                padded_attention_mask.append(cur_attention_mask)

        new_input_embeds = torch.stack(padded_input_embeds, dim=0)

        if labels is not None:
            new_labels = torch.stack(padded_labels, dim=0)
        else:
            new_labels = None

        if attention_mask is not None:
            attention_mask = torch.stack(padded_attention_mask, dim=0)
            assert attention_mask.shape == new_input_embeds.shape[:2]

        if labels is not None:
            assert new_labels.shape == new_input_embeds.shape[:2], (
                f"new_labels.shape={new_labels.shape}, "
                f"new_input_embeds.shape={new_input_embeds.shape}"
            )

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def compression_using_qformer(self, av_feat):
        Qformer = self.get_qformer()
        
        B, T, C = av_feat.size()
        av_feat_atts = torch.ones(av_feat.size()[:-1], dtype=torch.long).to(av_feat.device)
        av_feat = self.get_model().ln_head(av_feat)
        
        if self.config.window_level_Qformer:
            kernel = round(1500 * self.config.second_per_window / self.config.audio_frame_rate)
            stride = round(1500 * self.config.second_stride / self.config.audio_frame_rate)
            kernel = (1, kernel)
            stride = (1, stride)
            av_feat_tr = av_feat.transpose(1, 2).unsqueeze(2)
            av_feat_overlap = F.unfold(av_feat_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = av_feat_overlap.size()
            av_feat_overlap = av_feat_overlap.view(B, -1, kernel[1], L)
            av_feat_overlap = torch.permute(av_feat_overlap, [0, 3, 2, 1])
            av_feat = av_feat_overlap.reshape(-1, kernel[1], C)
            av_feat_atts = torch.ones(av_feat.size()[:-1], dtype=torch.long, device=av_feat.device)
        
        # Expand and slice query tokens: (B x max_length x token_dim)
        query_tokens = self.get_model().query_tokens.expand(av_feat.size(0), -1, -1)
        
        # Run Qformer (using its BERT) with cross attention to AV features
        query_output = Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=av_feat,
            encoder_attention_mask=av_feat_atts,
            return_dict=True
        )['last_hidden_state']

        query_output = query_output.view(B, -1, self.config.qformer_dim)
    
        return query_output