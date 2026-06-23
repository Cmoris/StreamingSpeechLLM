import torch
from torch import nn
from types import MethodType
from typing import Optional
from transformers.cache_utils import Cache

def patched_qwen2audio_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    """
    基本复制 Qwen2AudioForConditionalGeneration.forward，
    只在 language_model(...) 之前加：
        1. self.build_decoder_attention_mask(...)
        2. self.build_dual_audio_position_ids(...)
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    target_device = self.audio_tower.device

    if input_features is not None:
        input_features = input_features.to(target_device)
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(target_device)

    original_input_ids = input_ids
    original_attention_mask = attention_mask

    if inputs_embeds is None:
        # 1. Extract text embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and audios
        if input_features is not None and input_ids.shape[1] != 1:
            audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )

            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1

            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feat_lengths.dtype,
                    device=audio_feat_lengths.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.audio_tower.conv1.weight.dtype,
                device=self.audio_tower.conv1.weight.device,
            )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.multi_modal_projector(selected_audio_feature)

            # 判断是 legacy processing 还是 expanded audio tokens
            audio_tokens = input_ids == self.config.audio_token_id
            legacy_processing = (audio_tokens[:, :-1] & audio_tokens[:, 1:]).sum() == 0

            if legacy_processing:
                # 这个分支里 _merge 会返回 decoder 维度的 attention_mask 和 position_ids
                inputs_embeds, attention_mask, labels, position_ids, final_input_ids = (
                    self._merge_input_ids_with_audio_features(
                        audio_features,
                        audio_output_lengths,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        labels,
                    )
                )

                # legacy 分支下，input_ids 维度已经变了，要用 final_input_ids
                input_ids_for_hook = final_input_ids

            else:
                # processor 已经展开了 audio token，直接 masked_scatter
                num_audios, max_audio_tokens, embed_dim = audio_features.shape

                audio_features_mask = torch.arange(
                    max_audio_tokens,
                    device=audio_output_lengths.device,
                )[None, :]
                audio_features_mask = audio_features_mask < audio_output_lengths[:, None]
                audio_features = audio_features[audio_features_mask]

                n_audio_tokens = (input_ids == self.config.audio_token_id).sum().item()
                n_audio_features = audio_features.shape[0]

                if n_audio_tokens != n_audio_features:
                    raise ValueError(
                        f"Audio features and audio tokens do not match: "
                        f"tokens: {n_audio_tokens}, features {n_audio_features}"
                    )

                special_audio_mask = (input_ids == self.config.audio_token_id).to(inputs_embeds.device)
                special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

                input_ids_for_hook = input_ids

    else:
        input_ids_for_hook = input_ids
    breakpoint()
    # 1.channel embedding
    if hasattr(self, "add_dual_channel_embedding_to_inputs"):
        inputs_embeds = self.add_dual_channel_embedding_to_inputs(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids_for_hook,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
    breakpoint()
    # 2.position ids
    if hasattr(self, "build_dual_audio_position_ids"):
        position_ids = self.build_dual_audio_position_ids(
            input_ids=input_ids_for_hook,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
    breakpoint()
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        if attention_mask is not None:
            # 注意：如果你传入 4D attention_mask，这里的 loss mask 不能直接用 4D
            if attention_mask.ndim == 2:
                shift_attention_mask = attention_mask[..., 1:]
            else:
                # 4D attention mask 时，loss 仍然应该用 2D token mask
                # 所以建议额外保存 self._last_2d_decoder_attention_mask
                shift_attention_mask = self._last_2d_decoder_attention_mask[..., 1:]

            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # 保持和源码返回结构类似
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioCausalLMOutputWithPast

    return Qwen2AudioCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        attention_mask=attention_mask,
    )
