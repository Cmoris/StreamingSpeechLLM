from pathlib import Path
import json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel
from types import MethodType

from nonstreaming_data import DualChannelConvDataset, DualChannelASRDataset
from data_utils import read_audio, pad_audio_to_min_len, make_dummy_audio
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE, QUERY)


def build_one_turn_prompt(chunk_info, query="", sr=16000):
    MIN_AUDIO_SECS = 1.0
    min_samples = int(MIN_AUDIO_SECS * sr)

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
    for u in chunk_info["utterances"]:
        speaker = u["speaker"]
        text = u["text"].strip()
        if len(text) == 0:
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
        if u.get("is_turn_end", False):
            suffix += TE_TOKEN

        spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
        ref_parts.append(f"<{spk_tag}>{text}</{spk_tag}>{suffix}")

    ref_text = "".join(ref_parts)

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

    audio_inputs = [chunk_a, chunk_b]
    return conversation, audio_inputs, ref_text

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
            audio=[a.numpy() for a in audio_inputs],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
        )

        items.append({
            "inputs": inputs,
            "ref_text": ref_text,
        })

    return items

@torch.no_grad()
def generate_one_round_with_cache(
    model,
    processor,
    inputs,
    past_key_values=None,
    max_new_tokens=128,
):
    tokenizer = processor.tokenizer
    device = inputs["input_ids"].device

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 如果有历史 KV，需要拼 past attention mask
    if past_key_values is not None:
        past_len = past_key_values[0][0].shape[-2]
        past_attention_mask = torch.ones(
            (input_ids.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([past_attention_mask, attention_mask], dim=1)

    # 1. prefill 当前 user audio prompt
    outputs = model(
        **inputs,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = []

    # 2. decode assistant tokens
    for _ in range(max_new_tokens):
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

        # attention mask 增加一个位置
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
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() == tokenizer.convert_tokens_to_ids("<|im_end|>"):
            break

    pred_text = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    pred_text = pred_text.replace("<|im_end|>", "").strip()

    return pred_text, past_key_values


def run_generate_eval(
    base_model_name,
    lora_path,
    test_jsonl_dir,
    audio_root_a,
    audio_root_b,
    query,
    device="cuda:0",
    max_samples=None,
    max_new_tokens=128,
    output_path="predictions.jsonl",
):
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to(device)

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    ds = DualChannelConvDataset(
        annotation_paths=[str(p) for p in Path(test_jsonl_dir).glob("*.jsonl")],
        processor=processor,
        audio_root_a=audio_root_a,
        audio_root_b=audio_root_b,
        query=query,
    )

    ds.getitem_for_generate = MethodType(getitem_for_generate, ds)

    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(n)):
            gen_items = ds.getitem_for_generate(idx)
            past_key_values = None
            for turn_id, item in enumerate(gen_items):
                pred_text, past_key_values = generate_one_round_with_cache(
                    model=model,
                    processor=processor,
                    inputs=item["inputs"],
                    device=device,
                    max_new_tokens=max_new_tokens,
                    past_key_values=past_key_values
                )

                obj = {
                    "sample_id": idx,
                    "turn_id": turn_id,
                    "pred": pred_text,
                    "ref": item["ref_text"],
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    run_generate_eval(
        base_model_name="Qwen/Qwen2-Audio-7B-Instruct",
        lora_path="/ctd/Works/m-wu/Models/StreamingSpeechLLM/ASR_CONV_finetune/qwen2audio_l5_lora16",
        test_jsonl_dir="/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l3_conv_test_with_backchannel",
        audio_root_a="/ctd/Works/m-wu/Datasets/zoom2025/audios/A_gd",
        audio_root_b="/ctd/Works/m-wu/Datasets/zoom2025/audios/B_gd",
        query=QUERY,
        device="cuda:0",
        max_samples=None,
        max_new_tokens=256,
        output_path="results/predictions.jsonl",
    )