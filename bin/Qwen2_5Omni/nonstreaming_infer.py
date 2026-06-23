import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    HfArgumentParser,
)
from peft import PeftModel

from nonstreaming_data.conv_model_dataset import (
    DualChannelConvDataset,
    read_audio,
    make_dummy_audio,
    pad_audio_to_min_len,
)

from constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHUNK_SECS,
    DEFAULT_CONTEXT_LENGTH,
    TE_TOKEN,
    TS_TOKEN,
    SPEAKER_TOKENS,
    BC_TOKEN,
    PAUSE_TOKEN,
    SILENCE_TOKEN,
)


@dataclass
class DataArguments:
    data_version: int = field(default=2)
    annotation_dir: str = field(default="")
    audio_root_a: str = field(default="")
    audio_root_b: str = field(default="")
    output_dir: str = field(default="./infer_results")
    sample_rate: int = field(default=DEFAULT_SAMPLE_RATE)
    context_length: int = field(default=DEFAULT_CONTEXT_LENGTH)
    query: str = field(default="")
    max_new_tokens: int = field(default=128)


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(default="Qwen/Qwen2-Audio-7B-Instruct")
    model_path: str = field(
        default="/mnt/nvme/workspaces/muyun/Models/ASR_TS/qwen2audio_c2s_chunk1s_lora16/checkpoint-213000"
    )


# ===== CER =====
def compute_cer(ref, hyp):
    import numpy as np

    ref, hyp = list(ref), list(hyp)
    dp = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if ref[i - 1] == hyp[j - 1]
                else min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1,
                )
            )

    return dp[len(ref)][len(hyp)] / max(1, len(ref))


def move_to_device(inputs, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }


def get_past_len(past_key_values):
    if past_key_values is None:
        return 0

    # transformers 新版 Cache 对象
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()

    # legacy tuple: layer -> (key, value)
    return past_key_values[0][0].shape[-2]


def clean_generated_text(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def build_ref_text_from_chunk(chunk_info):
    """
    和你的 dataset/build_conversation 里的 assistant_text 构造逻辑保持一致。
    """
    parts = []

    if chunk_info["A"] is not None:
        start_time = chunk_info["A"]["audio_start"]
        end_time = chunk_info["A"]["audio_end"]
    else:
        start_time = chunk_info["B"]["audio_start"]
        end_time = chunk_info["B"]["audio_end"]

    for u in chunk_info["utterances"]:
        if u["end"] < start_time:
            continue
        if u["start"] > end_time:
            continue

        text = u["text"].strip()
        if not text:
            continue

        suffix = ""
        if u.get("is_turn_taking", False):
            suffix += TS_TOKEN
        if u.get("is_back_channel", False):
            suffix += BC_TOKEN
        if u.get("is_pause", False):
            suffix += PAUSE_TOKEN
        if u.get("is_silence", False):
            suffix += SILENCE_TOKEN

        spk_tag = "speaker_A" if u["speaker"] == "A" else "speaker_B"
        parts.append(f"<{spk_tag}>{text}</{spk_tag}>{suffix}")

    return "".join(parts)


def build_single_round_inputs(
    processor,
    chunk_info,
    query: str,
    sr: int,
    device,
):
    """
    只构造当前一轮：
      user: audio_A + audio_B + optional query
      assistant: 留空，由模型生成

    注意这里必须 add_generation_prompt=True。
    """
    min_audio_secs = 1.0
    min_samples = int(min_audio_secs * sr)

    a_info = chunk_info["A"]
    b_info = chunk_info["B"]

    if a_info is not None:
        start_time = a_info["audio_start"]
        end_time = a_info["audio_end"]
    else:
        start_time = b_info["audio_start"]
        end_time = b_info["audio_end"]

    if a_info is not None:
        chunk_a, _, _ = read_audio(a_info)
        chunk_a = pad_audio_to_min_len(chunk_a, min_samples)
    else:
        length = max(int((end_time - start_time) * sr), min_samples)
        chunk_a = make_dummy_audio(length)

    if b_info is not None:
        chunk_b, _, _ = read_audio(b_info)
        chunk_b = pad_audio_to_min_len(chunk_b, min_samples)
    else:
        length = max(int((end_time - start_time) * sr), min_samples)
        chunk_b = make_dummy_audio(length)

    user_content = [
        {"type": "audio", "audio": None},
        {"type": "audio", "audio": None},
    ]

    if query:
        user_content.append({"type": "text", "text": query})

    conversation = [
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        audio=[chunk_a.numpy(), chunk_b.numpy()],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    return move_to_device(inputs, device)


@torch.no_grad()
def generate_one_round_with_cache(
    model,
    processor,
    inputs,
    past_key_values=None,
    max_new_tokens: int = 128,
):
    """
    手写 greedy decode，原因是：
    1. 当前轮 prefill 时需要 audio inputs；
    2. 后续生成 token 时不再需要 input_features；
    3. 要把生成出的 assistant tokens 也更新进 KV cache，下一轮才能继续。
    """
    tokenizer = processor.tokenizer
    device = inputs["input_ids"].device

    input_ids = inputs["input_ids"]
    cur_attention_mask = inputs["attention_mask"]

    past_len = get_past_len(past_key_values)

    if past_len > 0:
        past_attention_mask = torch.ones(
            (input_ids.shape[0], past_len),
            dtype=cur_attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat(
            [past_attention_mask, cur_attention_mask],
            dim=1,
        )
    else:
        attention_mask = cur_attention_mask

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }

    # 只有当前 user turn 的 prefill 需要音频特征
    if "input_features" in inputs:
        model_inputs["input_features"] = inputs["input_features"]
    if "feature_attention_mask" in inputs:
        model_inputs["feature_attention_mask"] = inputs["feature_attention_mask"]

    outputs = model(**model_inputs)
    past_key_values = outputs.past_key_values

    next_token_logits = outputs.logits[:, -1, :]

    generated_ids = []

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(im_end_id)

    for _ in range(max_new_tokens):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_ids.append(token_id)

        # 把刚生成的 token 也喂回模型，更新 KV cache。
        # 即使 token 是 <|im_end|>，也最好写入 cache，
        # 这样下一轮 user turn 接在完整 assistant turn 后面。
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        if token_id in eos_ids:
            break

        next_token_logits = outputs.logits[:, -1, :]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    text = clean_generated_text(text)

    return text, past_key_values, generated_ids


def main():
    data_args, model_args = HfArgumentParser(
        (DataArguments, ModelArguments)
    ).parse_args_into_dataclasses()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ===== processor / tokenizer =====
    processor = AutoProcessor.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=True,
    )

    new_tokens = [
        TE_TOKEN,
        TS_TOKEN,
        BC_TOKEN,
        PAUSE_TOKEN,
        SILENCE_TOKEN,
        SPEAKER_TOKENS["A"][0],
        SPEAKER_TOKENS["A"][1],
        SPEAKER_TOKENS["B"][0],
        SPEAKER_TOKENS["B"][1],
    ]

    processor.tokenizer.add_tokens(new_tokens, special_tokens=False)

    # ===== 模型 =====
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)

    if model_args.model_path:
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_path,
        )
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()

    annotation_paths = [str(x) for x in Path(data_args.annotation_dir).glob("*.jsonl")]

    dataset = DualChannelConvDataset(
        annotation_paths=annotation_paths,
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        sample_rate=data_args.sample_rate,
        context_length=data_args.context_length,
        query=data_args.query,
    )

    Path(data_args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(data_args.output_dir) / f"results_rank{local_rank}.jsonl"

    # 避免 DistributedSampler padding 导致重复样本
    indices = list(range(local_rank, len(dataset), world_size))

    total_cer = 0.0
    total_rounds = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample_idx in indices:
            record = dataset.load_record(sample_idx)

            # record[0] 是 user 侧数据；你的 _resolve_audio 需要这个结构
            audio_list = dataset._resolve_audio(record[0])

            past_key_values = None
            round_results = []

            full_ref = []
            full_hyp = []

            for round_idx, chunk_info in enumerate(audio_list):
                inputs = build_single_round_inputs(
                    processor=processor,
                    chunk_info=chunk_info,
                    query=data_args.query or "",
                    sr=data_args.sample_rate,
                    device=device,
                )
                breakpoint()
                hyp_text, past_key_values, generated_ids = generate_one_round_with_cache(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    past_key_values=past_key_values,
                    max_new_tokens=data_args.max_new_tokens,
                )

                ref_text = build_ref_text_from_chunk(chunk_info)

                full_ref.append(ref_text)
                full_hyp.append(hyp_text)

                cer = compute_cer(ref_text, hyp_text) if ref_text else None

                round_results.append(
                    {
                        "round_idx": round_idx,
                        "ref": ref_text,
                        "hyp": hyp_text,
                        "cer": cer,
                    }
                )

                if cer is not None:
                    total_cer += cer
                    total_rounds += 1

            sample_ref = "".join(full_ref)
            sample_hyp = "".join(full_hyp)
            sample_cer = compute_cer(sample_ref, sample_hyp)

            out = {
                "sample_idx": sample_idx,
                "sample_cer": sample_cer,
                "ref": sample_ref,
                "hyp": sample_hyp,
                "rounds": round_results,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            if local_rank == 0 and total_rounds > 0 and sample_idx % 10 == 0:
                print(
                    f"[rank {local_rank}] sample={sample_idx}, "
                    f"sample_cer={sample_cer:.4f}, "
                    f"avg_round_cer={total_cer / total_rounds:.4f}"
                )

    dist.barrier()

    if local_rank == 0:
        print("Done.")
        if total_rounds > 0:
            print(f"Rank0 avg round CER: {total_cer / total_rounds:.4f}")


if __name__ == "__main__":
    main()