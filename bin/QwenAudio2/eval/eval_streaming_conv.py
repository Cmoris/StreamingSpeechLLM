import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen2AudioForConditionalGeneration,
)

QWEN_AUDIO_DIR = Path(__file__).resolve().parents[1]
if str(QWEN_AUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_AUDIO_DIR))

from constants import (  # noqa: E402
    BC_TOKEN,
    DEFAULT_CHUNK_SECS,
    DEFAULT_SAMPLE_RATE,
    PAUSE_TOKEN,
    QUERY,
    SILENCE_TOKEN,
    SPEAKER_TOKENS,
    TE_TOKEN,
    TS_TOKEN,
)
from data.conv_model_dataset import (  # noqa: E402
    DualChannelStreamingConvDataset,
    build_streaming_conversation,
)
from infer_utils import speaker_cer, special_token_f1_sequence  # noqa: E402


@dataclass
class DataArguments:
    annotation_dir: str = field(default="")
    audio_root_a: str = field(default="")
    audio_root_b: str = field(default="")
    output_path: str = field(default="streaming_conv_predictions.jsonl")
    chunk_secs: float = field(default=DEFAULT_CHUNK_SECS)
    sample_rate: int = field(default=DEFAULT_SAMPLE_RATE)
    query: str = field(default=QUERY)
    max_samples: Optional[int] = field(default=None)
    max_new_tokens: int = field(default=128)


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(default="Qwen/Qwen2-Audio-7B-Instruct")
    lora_path: Optional[str] = field(default=None)
    device: str = field(default="cuda:0")
    torch_dtype: str = field(default="float16")


def move_to_device(inputs, device):
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


def clean_generated_text(text: str) -> str:
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def evaluate_prediction(pred: str, ref: str) -> dict:
    return {
        "speaker_cer": speaker_cer(pred=pred, ref=ref),
        "special_token_f1": special_token_f1_sequence(pred=pred, ref=ref),
    }


def add_streaming_tokens(processor, model) -> None:
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
    added = processor.tokenizer.add_tokens(tokens, special_tokens=False)
    if added:
        model.resize_token_embeddings(len(processor.tokenizer))


def build_chunk_inputs(processor, user_msg, chunk_pair, sample_rate: int, device):
    chunk_a, chunk_b = chunk_pair
    conversation = [
        {
            "role": "user",
            "content": [
                dict(part) for part in user_msg["content"]
            ],
        }
    ]

    for part in conversation[0]["content"]:
        if part.get("type") != "audio":
            continue
        if part["audio"] == "__A__":
            part["audio"] = chunk_a.numpy()
        elif part["audio"] == "__B__":
            part["audio"] = chunk_b.numpy()

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        audio=[chunk_a.numpy(), chunk_b.numpy()],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    return move_to_device(inputs, device)


def build_generate_items(dataset: DualChannelStreamingConvDataset, index: int):
    record = dataset.load_record(index)
    stream = record[1]["content"][0]["text_stream"]
    audio_dict = dataset._resolve_audio(record[0])

    conversation, chunks_list = build_streaming_conversation(
        audio_dict["A"],
        audio_dict["B"],
        stream,
        dataset.query,
        chunk_secs=dataset.chunk_secs,
        sr=dataset.sr,
    )

    items = []
    chunk_idx = 0
    for msg_idx in range(0, len(conversation), 2):
        user_msg = conversation[msg_idx]
        assistant_msg = conversation[msg_idx + 1]
        items.append(
            {
                "user_msg": user_msg,
                "chunk_pair": chunks_list[chunk_idx],
                "ref_text": assistant_msg["content"][0]["text"],
            }
        )
        chunk_idx += 1

    return items


@torch.no_grad()
def generate_one_chunk_with_cache(
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
    return clean_generated_text(pred_text), past_key_values, generated_ids


def dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def run_generate_eval(data_args: DataArguments, model_args: ModelArguments):
    device = torch.device(model_args.device)
    processor = AutoProcessor.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=True,
    )

    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype=dtype_from_name(model_args.torch_dtype),
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    add_streaming_tokens(processor, base_model)

    if model_args.lora_path:
        model = PeftModel.from_pretrained(base_model, model_args.lora_path)
    else:
        model = base_model
    model.eval()

    annotation_paths = sorted(str(p) for p in Path(data_args.annotation_dir).glob("*.jsonl"))
    dataset = DualChannelStreamingConvDataset(
        annotation_paths=annotation_paths,
        processor=processor,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        chunk_secs=data_args.chunk_secs,
        sample_rate=data_args.sample_rate,
        query=data_args.query,
        eval=True,
    )

    output_path = Path(data_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(dataset)
    if data_args.max_samples is not None:
        n_samples = min(n_samples, data_args.max_samples)

    with output_path.open("w", encoding="utf-8") as fout:
        for sample_idx in tqdm(range(n_samples)):
            items = build_generate_items(dataset, sample_idx)
            past_key_values = None
            rounds = []
            full_ref = []
            full_pred = []

            for chunk_idx, item in enumerate(items):
                inputs = build_chunk_inputs(
                    processor=processor,
                    user_msg=item["user_msg"],
                    chunk_pair=item["chunk_pair"],
                    sample_rate=data_args.sample_rate,
                    device=device,
                )
                pred_text, past_key_values, generated_ids = generate_one_chunk_with_cache(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    past_key_values=past_key_values,
                    max_new_tokens=data_args.max_new_tokens,
                )

                ref_text = item["ref_text"]
                full_ref.append(ref_text)
                full_pred.append(pred_text)
                rounds.append(
                    {
                        "chunk_idx": chunk_idx,
                        "ref": ref_text,
                        "pred": pred_text,
                        "metrics": evaluate_prediction(pred_text, ref_text),
                        "generated_ids": generated_ids,
                    }
                )

            sample_ref = "".join(full_ref)
            sample_pred = "".join(full_pred)
            fout.write(
                json.dumps(
                    {
                        "sample_idx": sample_idx,
                        "ref": sample_ref,
                        "pred": sample_pred,
                        "metrics": evaluate_prediction(sample_pred, sample_ref),
                        "rounds": rounds,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()

    print(f"Saved to {output_path}")


def main():
    data_args, model_args = HfArgumentParser(
        (DataArguments, ModelArguments)
    ).parse_args_into_dataclasses()
    run_generate_eval(data_args, model_args)


if __name__ == "__main__":
    main()
