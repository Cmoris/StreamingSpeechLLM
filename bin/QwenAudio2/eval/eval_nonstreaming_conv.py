import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen2AudioForConditionalGeneration,
)

from constants import (  
    DEFAULT_SAMPLE_RATE,
    BC_TOKEN,
    PAUSE_TOKEN,
    QUERY,
    SILENCE_TOKEN,
    SPEAKER_TOKENS,
    TE_TOKEN,
    TS_TOKEN,
)
from data_utils import make_dummy_audio, pad_audio_to_min_len, read_audio  
from nonstreaming_data import DualChannelConvDataset  
from infer_utils import speaker_cer, special_token_f1_sequence  


@dataclass
class DataArguments:
    annotation_dir: str = field(default="")
    audio_root_a: str = field(default="")
    audio_root_b: str = field(default="")
    output_path: str = field(default="predictions.jsonl")
    query: str = field(default=QUERY)
    max_samples: Optional[int] = field(default=None)
    max_new_tokens: int = field(default=128)


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(default="Qwen/Qwen2-Audio-7B-Instruct")
    lora_path: Optional[str] = field(default=None)
    device: str = field(default="cuda:0")
    dtype: str = field(default="bfloat16")


def init_distributed(model_args: ModelArguments):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(model_args.device)

    return rank, world_size, local_rank, device


def rank_output_path(output_path: str, rank: int, world_size: int) -> Path:
    path = Path(output_path)
    if world_size <= 1:
        return path
    return path.with_name(f"{path.stem}_rank{rank}{path.suffix}")


def move_to_device(inputs: dict, device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def get_past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[-2]


def build_one_turn_prompt(chunk_info, query: str = "", sr: int = 16000):
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

    ref_parts = []
    for utterance in chunk_info["utterances"]:
        speaker = utterance["speaker"]
        text = utterance["text"].strip()
        if not text:
            continue

        suffix = ""
        if utterance.get("is_turn_taking", False):
            suffix += TS_TOKEN
        if utterance.get("is_back_channel", False):
            suffix += BC_TOKEN
        if utterance.get("is_pause", False):
            suffix += PAUSE_TOKEN
        if utterance.get("is_silence", False):
            suffix += SILENCE_TOKEN
        if utterance.get("is_turn_end", False):
            suffix += TE_TOKEN

        spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
        ref_parts.append(f"<{spk_tag}>{text}</{spk_tag}>{suffix}")

    conversation = [
        {"role": "system", "content": query},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Speaker A"},
                {"type": "audio", "audio": None},
                {"type": "text", "text": "Speaker B"},
                {"type": "audio", "audio": None},
            ],
        },
    ]

    return conversation, [chunk_a, chunk_b], "".join(ref_parts)


def getitem_for_generate(self, index: int):
    record = self.load_record(index)
    audio_list = self._resolve_audio(record[0])

    items = []
    for chunk_info in audio_list:
        conversation, audio_inputs, ref_text = build_one_turn_prompt(
            chunk_info,
            query=self.query,
            sr=self.sr,
        )

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            audio=[audio.numpy() for audio in audio_inputs],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
        )

        items.append(
            {
                "inputs": inputs,
                "ref_text": ref_text,
            }
        )

    return items


def clean_generated_text(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()

def evaluate_prediction(pred: str, ref: str) -> dict:
    return {
        "speaker_cer": speaker_cer(pred=pred, ref=ref),
        "special_token_f1": special_token_f1_sequence(pred=pred, ref=ref),
    }

def dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def add_streaming_tokens(processor) -> None:
    tokens = [
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
    
    processor.tokenizer.add_tokens(tokens, special_tokens=False)


@torch.no_grad()
def generate_one_round_with_cache(
    model,
    processor,
    inputs,
    past_key_values=None,
    max_new_tokens: int = 128,
):
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
        attention_mask = torch.cat([past_attention_mask, cur_attention_mask], dim=1)
    else:
        attention_mask = cur_attention_mask

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    if "input_features" in inputs:
        model_inputs["input_features"] = inputs["input_features"]
    if "feature_attention_mask" in inputs:
        model_inputs["feature_attention_mask"] = inputs["feature_attention_mask"]

    outputs = model(**model_inputs)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.add(im_end_id)

    generated_ids = []
    for _ in range(max_new_tokens):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_ids.append(token_id)

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

    pred_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return clean_generated_text(pred_text), past_key_values


def run_generate_eval(
    data_args: DataArguments,
    model_args: ModelArguments,
):
    rank, world_size, _, device = init_distributed(model_args)
    processor = AutoProcessor.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=True,
    )

    add_streaming_tokens(processor)

    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path,
        dtype=dtype_from_name(model_args.dtype),
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    if model_args.lora_path:
        model = PeftModel.from_pretrained(base_model, model_args.lora_path)
    else:
        model = base_model
    model.eval()

    dataset = DualChannelConvDataset(
        annotation_paths=sorted(str(path) for path in Path(data_args.annotation_dir).glob("*.jsonl")),
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        query=data_args.query,
    )
    dataset.getitem_for_generate = getitem_for_generate.__get__(dataset)

    n_samples = len(dataset)
    if data_args.max_samples is not None:
        n_samples = min(n_samples, data_args.max_samples)

    indices = list(range(rank, n_samples, world_size))
    output_path = rank_output_path(data_args.output_path, rank, world_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for sample_idx in tqdm(indices, disable=rank != 0):
            gen_items = dataset.getitem_for_generate(sample_idx)
            past_key_values = None

            for turn_idx, item in enumerate(gen_items):
                inputs = move_to_device(item["inputs"], device)
                pred_text, past_key_values = generate_one_round_with_cache(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=data_args.max_new_tokens,
                    past_key_values=past_key_values,
                )
                ref_text = item["ref_text"]
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_idx,
                            "turn_id": turn_idx,
                            "rank": rank,
                            "pred": pred_text,
                            "ref": ref_text,
                            "metrics": evaluate_prediction(pred_text, ref_text),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    print(f"Saved to {output_path}")


def main():
    data_args, model_args = HfArgumentParser(
        (DataArguments, ModelArguments)
    ).parse_args_into_dataclasses()
    run_generate_eval(data_args, model_args)


if __name__ == "__main__":
    main()
