import torch
from torch import nn
from types import MethodType
from typing import Optional
from transformers.cache_utils import Cache


def build_dual_audio_position_ids_from_input_ids(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    inputs_embeds: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
):
    """
    让同一个样本里的两个 audio span 使用相同 position ids。

    适用于 processor 已经把 <|AUDIO|> 展开成连续多个 audio token 的情况：
        ... <|AUDIO|> <|AUDIO|> ... <|AUDIO|> ...
        ... <|AUDIO|> <|AUDIO|> ... <|AUDIO|> ...

    即源码里的 non-legacy 分支。
    """

    device = inputs_embeds.device
    B, T = input_ids.shape

    if attention_mask is None:
        attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # 默认 position_ids
    if position_ids is None:
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
    else:
        position_ids = position_ids.to(device).clone()

    audio_token_id = model.config.audio_token_id
    audio_mask = (input_ids.to(device) == audio_token_id) & (attention_mask == 1)

    for b in range(B):
        idx = torch.where(audio_mask[b])[0]
        if idx.numel() == 0:
            continue

        # 找连续 audio span
        # spans: [(start, end), ...), end 为 exclusive
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

        # 如果一个样本里有两个音频，就让第 2 个音频的 position ids 对齐第 1 个音频
        # 如果超过两个，也都对齐第一个。
        if len(spans) >= 2:
            ref_s, ref_e = spans[0]
            ref_len = ref_e - ref_s

            for s, e in spans[1:]:
                cur_len = e - s
                n = min(ref_len, cur_len)

                # 两个音频共同部分使用相同位置
                position_ids[b, s : s + n] = position_ids[b, ref_s : ref_s + n]

                # 如果第二个音频更长，后面继续递增
                if cur_len > n:
                    last = position_ids[b, s + n - 1]
                    extra = torch.arange(
                        1,
                        cur_len - n + 1,
                        device=device,
                        dtype=position_ids.dtype,
                    )
                    position_ids[b, s + n : e] = last + extra

    return position_ids


