import json
import os
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
    BC_TOKEN,
    DEFAULT_SAMPLE_RATE,
    PAUSE_TOKEN,
    QUERY,
    SILENCE_TOKEN,
    SPEAKER_TOKENS,
    TE_TOKEN,
    TS_TOKEN,
)
from data_utils import make_dummy_audio, pad_audio_to_min_len, read_audio
from infer_utils import speaker_cer, special_token_f1_sequence
from nonstreaming_data import DualChannelConvDataset


@dataclass
class DataArguments:
    annotation_dir: str = field(default="")
    audio_root_a: str = field(default="")
    audio_root_b: str = field(default="")
    output_path: str = field(default="incremental_generate_predictions.jsonl")
    query: str = field(default=QUERY)
    sample_rate: int = field(default=DEFAULT_SAMPLE_RATE)
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


def clean_generated_text(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def evaluate_prediction(pred: str, ref: str) -> dict:
    return {
        "speaker_cer": speaker_cer(pred=pred, ref=ref),
        "special_token_f1": special_token_f1_sequence(pred=pred, ref=ref),
    }


def build_incremental_turn(chunk_info, sr: int = DEFAULT_SAMPLE_RATE):
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

    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Speaker A"},
            {"type": "audio", "audio": None},
            {"type": "text", "text": "Speaker B"},
            {"type": "audio", "audio": None},
        ],
    }
    return user_msg, [chunk_a, chunk_b], "".join(ref_parts)


def build_incremental_items(dataset: DualChannelConvDataset, index: int):
    record = dataset.load_record(index)
    audio_list = dataset._resolve_audio(record[0])

    items = []
    for chunk_info in audio_list:
        user_msg, audio_inputs, ref_text = build_incremental_turn(
            chunk_info,
            sr=dataset.sr,
        )
        items.append(
            {
                "user_msg": user_msg,
                "audio_inputs": audio_inputs,
                "ref_text": ref_text,
            }
        )
    return items


def build_processor_inputs(processor, conversation, audio_inputs, sample_rate: int, device):
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=text,
        audio=[audio.numpy() for audio in audio_inputs],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    return move_to_device(inputs, device)


@torch.no_grad()
def generate_one_turn(model, processor, inputs, max_new_tokens: int = 128):
    tokenizer = processor.tokenizer
    input_len = inputs["input_ids"].size(1)
    eos_token_id = []
    if tokenizer.eos_token_id is not None:
        eos_token_id.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_token_id.append(im_end_id)

    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if eos_token_id:
        generate_kwargs["eos_token_id"] = eos_token_id
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    generated = model.generate(**generate_kwargs)
    generated = generated[:, input_len:]
    pred_text = tokenizer.decode(
        generated[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return clean_generated_text(pred_text)


def run_incremental_generate_eval(
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
        torch_dtype=dtype_from_name(model_args.dtype),
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    
    if model_args.lora_path:
        model = PeftModel.from_pretrained(base_model, model_args.lora_path)
    else:
        model = base_model
    model.eval()

    annotation_paths = sorted(str(path) for path in Path(data_args.annotation_dir).glob("*.jsonl"))
    dataset = DualChannelConvDataset(
        annotation_paths=annotation_paths,
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        sample_rate=data_args.sample_rate,
        query=data_args.query,
    )

    n_samples = len(dataset)
    if data_args.max_samples is not None:
        n_samples = min(n_samples, data_args.max_samples)

    indices = list(range(rank, n_samples, world_size))
    output_path = rank_output_path(data_args.output_path, rank, world_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for sample_idx in tqdm(indices, disable=rank != 0):
            items = build_incremental_items(dataset, sample_idx)
            conversation = [{"role": "system", "content": data_args.query}]
            audio_inputs = []
            rounds = []
            full_ref = []
            full_pred = []

            for turn_idx, item in enumerate(items):
                conversation.append(item["user_msg"])
                audio_inputs.extend(item["audio_inputs"])

                inputs = build_processor_inputs(
                    processor=processor,
                    conversation=conversation,
                    audio_inputs=audio_inputs,
                    sample_rate=data_args.sample_rate,
                    device=device,
                )
                pred_text = generate_one_turn(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=data_args.max_new_tokens,
                )
                conversation.append({"role": "assistant", "content": pred_text})

                ref_text = item["ref_text"]
                full_ref.append(ref_text)
                full_pred.append(pred_text)
                rounds.append(
                    {
                        "turn_id": turn_idx,
                        "pred": pred_text,
                        "ref": ref_text,
                        "metrics": evaluate_prediction(pred_text, ref_text),
                    }
                )

            sample_ref = "".join(full_ref)
            sample_pred = "".join(full_pred)
            fout.write(
                json.dumps(
                    {
                        "sample_id": sample_idx,
                        "rank": rank,
                        "pred": sample_pred,
                        "ref": sample_ref,
                        "metrics": evaluate_prediction(sample_pred, sample_ref),
                        "rounds": rounds,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    print(f"Saved to {output_path}")


def main():
    data_args, model_args = HfArgumentParser(
        (DataArguments, ModelArguments)
    ).parse_args_into_dataclasses()
    run_incremental_generate_eval(data_args, model_args)


if __name__ == "__main__":
    main()
