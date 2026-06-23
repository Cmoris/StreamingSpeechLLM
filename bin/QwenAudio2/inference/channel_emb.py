import torch
from torch import nn


def add_channel_embedding(model, num_channels=2):
    hidden_size = model.config.text_config.hidden_size

    model.audio_channel_embedding = nn.Embedding(num_channels, hidden_size)

    # 初始化得小一点，避免一开始破坏 pretrained audio/text embedding 空间
    nn.init.normal_(model.audio_channel_embedding.weight, mean=0.0, std=0.02)

    model.audio_channel_embedding.to(
        device=model.language_model.device,
        dtype=model.get_input_embeddings().weight.dtype,
    )

    return model

def add_dual_channel_embedding_to_inputs(
    self,
    inputs_embeds,
    input_ids,
    attention_mask=None,
    **kwargs,
):
    """
    给每个样本里的两个 audio span 加 channel embedding。

    假设：
        第 1 个连续 audio span 是 A channel
        第 2 个连续 audio span 是 B channel

    inputs_embeds: [B, T, H]
    input_ids:     [B, T]
    """

    if input_ids is None:
        return inputs_embeds

    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    input_ids = input_ids.to(device)

    if attention_mask is None:
        attention_mask = torch.ones(
            input_ids.shape,
            dtype=torch.long,
            device=device,
        )
    else:
        attention_mask = attention_mask.to(device)

    audio_mask = (input_ids == self.config.audio_token_id) & (attention_mask == 1)

    channel_embed = self.audio_channel_embedding.to(
        device=device,
        dtype=dtype,
    )

    inputs_embeds = inputs_embeds.clone()

    B = input_ids.shape[0]

    for b in range(B):
        idx = torch.where(audio_mask[b])[0]
        if idx.numel() == 0:
            continue

        # 找连续 audio span
        spans = []
        start = idx[0].item()
        prev = idx[0].item()

        for x in idx[1:].tolist():
            if x == prev + 1:
                prev = x
            else:
                spans.append((start, prev + 1))
                start = x
                prev = x

        spans.append((start, prev + 1))

        # 按 A, B, A, B... 顺序分配 channel id
        for i, (s, e) in enumerate(spans):
            channel_id = i % 2

            ch = torch.tensor(
                channel_id,
                device=device,
                dtype=torch.long,
            )

            inputs_embeds[b, s:e, :] = inputs_embeds[b, s:e, :] + channel_embed(ch)

    return inputs_embeds