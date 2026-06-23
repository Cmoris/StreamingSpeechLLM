from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import  (
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3Config,
    AutoConfig, 
    AutoModelForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .speechllm_model import AudioMetaForCausalLM, AudioMetaModel

class SpeechQwen3Config(Qwen3Config):
    model_type = "speech_qwen3"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
class SpeechQwen3Model(AudioMetaModel, Qwen3Model):
    config_class = SpeechQwen3Config
    
    def __init__(self, config):
        super(SpeechQwen3Model, self).__init__(config)
        

class SpeechQwen3ForCausalLM(Qwen3ForCausalLM, AudioMetaForCausalLM):
    config_class = SpeechQwen3Config
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = SpeechQwen3Model(config)
        # self.vocab_size = config.vocab_size
        # self.hidden_size = config.hidden_size
        # self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        self.post_init()
        
    def get_model(self):
        return self.model
    
    # CORRECTED forward METHOD
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        audios: Optional[dict] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # The multimodal preparation logic has been removed from here.
        # This method is now a clean passthrough, compatible with the generate loop.
        
        # We can add logic to handle a `source` kwarg for training/inference outside of `generate`
        
        if past_key_values is None and inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                audios
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audios: Optional[dict] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if audios is not None:
            
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=None,
                audios=audios
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
AutoConfig.register("speech_qwen3", SpeechQwen3Config)
AutoModelForCausalLM.register(SpeechQwen3Config, SpeechQwen3ForCausalLM)
