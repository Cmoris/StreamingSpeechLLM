import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, AutoFeatureExtractor, logging

import sys
sys.path.append("../")
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE, DEFAULT_CONTEXT_LENGTH)
from data_utils import read_audio, safe_chunk, make_dummy_audio, pad_audio_to_min_len

logger = logging.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_conversation(
    audio_list,
    query="",
    sr=16000,
):
    MIN_AUDIO_SECS = 1.0
    min_samples = int(MIN_AUDIO_SECS * sr)
    
    conversation = []
    audio_inputs = []

    for chunk_info in audio_list:
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

        cur_uttrs = []

        for u in chunk_info["utterances"]:
            if u["end"] < start_time:
                continue
            if u["start"] > end_time:
                continue

            speaker = u["speaker"]
            text = u["text"].strip()
            if len(text) == 0:
                continue

            spk_tag = "speaker_A" if speaker == "A" else "speaker_B"
            cur_uttrs.append(f"<{spk_tag}>{text}</{spk_tag}>")

        assistant_text = "".join(cur_uttrs)
        
        user_content = []
        
        if query:
            user_content.append({"type": "text", "text": query})

        user_content.extend([
            {"type": "text", "text": "Speaker A"},
            {"type": "audio", "audio": None},
            {"type": "text", "text": "Speaker B"},
            {"type": "audio", "audio": None},
        ])

        audio_inputs.append(chunk_a)
        audio_inputs.append(chunk_b)

        
        conversation.append({"role": "user", "content": user_content})
        conversation.append({"role": "assistant", "content": assistant_text})

    return conversation, audio_inputs



class DualChannelASRDataset(Dataset):
    # SYSTEM_PROMPT = (
    #     "You are a real-time dual-channel meeting transcriber. "
    #     "For each audio chunk you receive, extend the running transcript "
    #     "using <speaker_A>&</speaker_A>/<speaker_B>&</speaker_B> speaker tags with <ts> (turn-start) and <te> (turn-end) tokens."
    # )

    # QUERY = (
    #     "Transcribe the conversation between the two speakers in real time. "
    #     "Use [A] and [B] tags, <ts> and <te> tokens."
    # )
    SYSTEM_PROMPT = ""

    QUERY = ""

    def __init__(
        self,
        annotation_paths: list[str],
        processor: Optional[AutoProcessor],
        audio_root_a: str,
        audio_root_b: str,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
    ):
        super().__init__()

        self.processor      = processor
        self.audio_root_a   = Path(audio_root_a)
        self.audio_root_b   = Path(audio_root_b)
        self.context_length = context_length
        self.sr             = sample_rate
        self.query          = query or self.QUERY
        # ── special token ids for label masking  ──────────────
        (
            self.im_start_id,
            self.assistant_id,
            self.newline_id,
            self.im_end_id,
        ) = processor.tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids

        # ── build seek-based handle list (same as LiveCC) ────────────────────
        self.handles: list[tuple[str, int]] = []
        for ap in annotation_paths:
            ap = str(ap)
            if ap.endswith(".jsonl"):
                # last line stores seek indices
                seeks = json.loads(_read_last_line(ap))
                self.handles.extend([(ap, sk) for sk in seeks])
                # logger.warning(f"Loaded {ap} ({len(seeks)} samples)")
            elif ap.endswith(".json"):
                # single-record JSON; seek=0 sentinel handled in load_record
                self.handles.append((ap, -1))
                logger.warning(f"Loaded single-record {ap}")
            else:
                raise ValueError(f"Unsupported annotation format: {ap}")

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def load_record(self, index: int) -> dict:
        path, seek = self.handles[index]
        if seek == -1:                          # single .json file
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        with open(path, encoding="utf-8") as f:
            f.seek(seek)
            return json.loads(f.readline())
        
    def _resolve_audio(self, ele: dict) -> list[dict]:
        utterances = ele["content"][0]["utterances"]
        conv_id = ele["content"][1]["id"]

        audio_a_path = str(self.audio_root_a / f"{conv_id}_a.wav")
        audio_b_path = str(self.audio_root_b / f"{conv_id}_b.wav")

        audio_list = []

        utterances = sorted(utterances, key=lambda x: (x["start"], x["end"]))

        for u in utterances:
            t_start = float(u["start"])
            t_end = float(u["end"])

            if t_end <= t_start:
                continue

            audio_dict = {
                "A": {
                    "audio": audio_a_path,
                    "audio_start": t_start,
                    "audio_end": t_end,
                },
                "B": {
                    "audio": audio_b_path,
                    "audio_start": t_start,
                    "audio_end": t_end,
                },
                "utterances": [u],
            }

            audio_list.append(audio_dict)

        return audio_list

    # ── Core item builder ────────────────────────────────────────────────────

    def getitem(self, index: int) -> dict:
        record   = self.load_record(index)
        
        utterances = record[0]['content'][0]["utterances"]
        
        audio_list = self._resolve_audio(record[0])
        
        # ── build streaming multi-turn conversation ───────────────────────────
        conversation, audio_inputs = build_conversation(
            audio_list, self.query, sr=self.sr,
        )                    
        # ── tokenize ─────────────────────────────────────────────────────────
        
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=text,
            audio=[a.numpy() for a in audio_inputs],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True,
        )
        
        input_ids = inputs["input_ids"]                              # (1, L)
        
        # ── label masking: only assistant turns are supervised ────────────────
        labels = torch.full_like(input_ids, fill_value=-100)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs   = (input_ids == self.im_end_id  ).nonzero()

        for (si, s_idx), (_, e_idx) in zip(im_start_idxs, im_end_idxs):
            # +1 skips <|im_start|>, check next token == "assistant"
            if input_ids[si, s_idx + 1] == self.assistant_id:
                # im_start | assistant | \n | <tokens> | im_end
                #   +0          +1       +2     +3 …      e_idx
                labels[si, s_idx + 3 : e_idx + 1] = \
                    input_ids[si, s_idx + 3 : e_idx + 1]
    
        inputs["labels"] = labels

        return inputs

    # ── Public Dataset API ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, index: int) -> dict:
        max_tries = 10
        for _ in range(max_tries):
            try:
                return self.getitem(index)
            except Exception as e:
                logger.warning(f"Failed {_}-th try to get item {index}: {e}")
                index = random.randint(0, self.__len__() - 1)
                logger.warning(f"Retrying to get item {index}")
        raise Exception(f"Failed to get item after {max_tries} retries")
        

    def data_collator(self, batched_inputs: list[dict], **kwargs) -> dict:
        """
        LiveCC-style collator: batch_size=1 only (multi-turn audio is huge).
        For batch_size>1 override this with a proper padded collator.
        """
        assert len(batched_inputs) == 1, \
            "Use batch_size=1 for streaming audio; override data_collator for larger batches."
        return batched_inputs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: read last line efficiently (mirror LiveCC's readlastline)
# ─────────────────────────────────────────────────────────────────────────────

def _read_last_line(path: str, buf: int = 4096) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        pos, last = size, b""
        while pos > 0:
            read_sz = min(buf, pos)
            pos -= read_sz
            f.seek(pos)
            chunk = f.read(read_sz)
            lines = (chunk + last).split(b"\n")
            last  = lines[0]
            non_empty = [l for l in lines[1:] if l.strip()]
            if non_empty:
                return non_empty[-1].decode("utf-8")
    return last.decode("utf-8")


def record_display(record: dict):
    print("=== Utterances ===")
    for u in record[0]["content"][0]["utterances"]:
        flag = " ← TURN-TAKING" if u["is_turn_taking"] else ""
        print(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{flag}")

    print("\n=== Stream (first 20 events) ===")
    for ev in record[1]["content"][0]["text_stream"]:
        print(f"  {ev['start']:.3f} {ev['end']:.3f}  [{ev['speaker']}]  {ev['token']:6s}  ({ev['kind']})")
    print("\n=== Training sequence ===")

# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    from torch.utils.data import DataLoader
    import tqdm
    
    logger = logging.getLogger(__name__)

    # 配置 logger
    logging.basicConfig(
        filename="debug.log",
        filemode="w",  # 每次运行覆盖；改成 "a" 是追加
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )
    
    dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel"

    from transformers import Qwen2AudioProcessor, AutoProcessor, Qwen2AudioForConditionalGeneration
    proc = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
    )
    audio_token_id = proc.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    ds = DualChannelASRDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=proc,
        audio_root_a="/n/work6/yizhang/Moris/zoom2025/audios/A_gd",
        audio_root_b="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
        context_length=2,
    )
    print(f"Dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=16, collate_fn=ds.data_collator)

    device = "cuda:4"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.float16,
        device_map=None,
    ).to(device)

    model.eval()

    for i, batch in enumerate(tqdm.tqdm(loader)):
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        try:
            with torch.no_grad():
                outputs = model(**batch)
        except Exception as e:
            print("=" * 80)
            print("BAD BATCH:", i)
            print("error:", repr(e))

            audio_token_id = model.config.audio_token_id
            input_ids = batch["input_ids"]

            print("input_ids:", input_ids.shape)
            print("input_features:", batch["input_features"].shape)
            print("attention_mask:", batch["attention_mask"].shape)
            print("feature_attention_mask:", batch["feature_attention_mask"].shape)

            print("total <|AUDIO|> token ids:",
                (input_ids == audio_token_id).sum().item())

            print("decoded text:")
            print(proc.tokenizer.decode(
                input_ids[0].detach().cpu(),
                skip_special_tokens=False,
            ))

            raise
        # Multimodal_tokens = (batch["input_ids"] == (audio_token_id+1)).sum().item()
        # logging.info(f"Multimodal tokens in input_ids: {Multimodal_tokens}")
        # logging.info(f"num input features: {batch['input_features'].shape[0]}")
        # # logging.info(f"text: {batch['text']}")
        # # logging.info(f"input_ids : {batch['input_ids'].size()}")
        # # logging.info(f"labels: {batch['labels'].size()}")
        # # logging.info(f"audios : {batch['input_features'].size()}")
        # # logging.info(f"original audio sizes: {batch['original_audio_size']}")
        # target_ids = batch["input_ids"]
        # target_ids[target_ids == -100] = proc.tokenizer.pad_token_id
        # refs = proc.batch_decode(target_ids, skip_special_tokens=True)
        # logging.info(f"texts : {refs}")