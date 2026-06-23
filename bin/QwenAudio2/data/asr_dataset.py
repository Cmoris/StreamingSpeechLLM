import re
import json
import random
from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, logging
import sys
sys.path.append("../")

from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN, QUERY,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE)
from data_utils import read_audio, safe_chunk, make_dummy_audio, pad_audio_to_min_len


logger = logging.get_logger(__name__)


def get_transcript_before_time(
    stream: list[dict],
    t_end: float,
    start_from: int = 0,
    eps: float = 1e-6,
) -> tuple[str, int]:
    """
    Serialize stream events whose end time <= t_end.

    Returns only NEW events from start_from to current t_end.
    """
    parts: list[str] = []
    i = start_from

    current_speaker: str | None = None

    def close_speaker():
        nonlocal current_speaker
        if current_speaker is not None:
            parts.append(SPEAKER_TOKENS[current_speaker][1])
            current_speaker = None

    def open_speaker(spk: str):
        nonlocal current_speaker
        if current_speaker != spk:
            close_speaker()
            parts.append(SPEAKER_TOKENS[spk][0])
            current_speaker = spk

    while i < len(stream) and stream[i]["end"] <= t_end + eps:
        ev = stream[i]
        kind = ev.get("kind")
        spk = ev.get("speaker")
        token = ev.get("token", "")

        if kind == "asr":
            open_speaker(spk)
            parts.append(token)

        elif kind == "bc":
            close_speaker()
            parts.append(BC_TOKEN if isinstance(BC_TOKEN, str) else BC_TOKEN[0])

        elif kind == "silence":
            close_speaker()
            parts.append(SILENCE_TOKEN)

        elif kind == "pause":
            close_speaker()
            parts.append(PAUSE_TOKEN)

        elif kind == "ts":
            close_speaker()
            parts.append(TS_TOKEN)

        elif kind == "te":
            close_speaker()
            parts.append(TE_TOKEN)

        else:
            # 未知 kind，保守跳过
            pass

        i += 1

    close_speaker()

    return "".join(parts), i


# ─────────────────────────────────────────────────────────────────────────────
# Streaming conversation builder  (core of LiveCC adaptation)
# ─────────────────────────────────────────────────────────────────────────────


def build_streaming_conversation(
    audio: dict | None,
    stream: list[dict],
    query: str,
    chunk_secs: float = DEFAULT_CHUNK_SECS,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[list[dict], list[tuple[torch.Tensor, torch.Tensor]]]:

    chunk_samples = int(chunk_secs * sr)

    wav, _, _ = read_audio(audio) if audio else (
        make_dummy_audio(chunk_samples), np.array([]), None
    )

    total_samples = len()

    # stream 里面的时间是全局时间，因此 chunk 的相对时间需要加 offset
    if audio is not None:
        offset = audio.get("audio_start", 0.0)

    conversation: list[dict] = []
    chunks_list: list[tuple[torch.Tensor, torch.Tensor]] = []

    next_start_from = 0
    first_turn = True

    for chunk_start in range(0, total_samples, chunk_samples):
        chunk_end = chunk_start + chunk_samples

        t_start = chunk_start / sr
        t_end = chunk_end / sr

        abs_t_end = offset + t_end

        chunk = safe_chunk(wav, chunk_start, chunk_end, chunk_samples)

        phrase, next_start_from = get_transcript_before_time(
            stream=stream,
            t_end=abs_t_end,
            start_from=next_start_from,
        )

        user_content = [
            {"type": "audio", "audio": None},
        ]

        if first_turn:
            user_content.insert(0, {"type": "text", "text": query})
            first_turn = False

        # 没有新输出时，用 streaming continuation token 占位
        assistant_text = phrase if phrase else STREAMING_CONT

        conversation.append({
            "role": "user",
            "content": user_content,
        })

        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_text,
                }
            ],
        })

        chunks_list.append(chunk)

    while len(conversation) >= 2:
        last_text = conversation[-1]["content"][0]["text"]
        if last_text == STREAMING_CONT:
            conversation = conversation[:-2]
            chunks_list = chunks_list[:-1]
        else:
            break

    return conversation, chunks_list


class SingleChannelStreamingASRDataset(Dataset):
    # SYSTEM_PROMPT = (
    #     "You are a real-time dual-channel meeting transcriber. "
    #     "For each audio chunk you receive, extend the running transcript "
    #     "using [A]/[B] speaker tags with <ts> (turn-start) and <te> (turn-end) tokens."
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
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
    ):
        super().__init__()

        self.processor      = processor
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
                last = _read_last_line(ap)
                use_seek_footer = False
                try:
                    seeks = json.loads(last)
                    use_seek_footer = (
                        isinstance(seeks, list)
                        and all(isinstance(x, int) for x in seeks)
                    )
                except Exception:
                    use_seek_footer = False

                if use_seek_footer:
                    self.handles.extend([(ap, sk) for sk in seeks])
                else:
                    with open(ap, encoding="utf-8") as f:
                        while True:
                            sk = f.tell()
                            line = f.readline()
                            if not line:
                                break
                            if line.strip():
                                self.handles.append((ap, sk))
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
        
    def _resolve_audio(self, ele: dict) -> tuple[Path, Path]:
        utterances = ele['content'][0]["utterances"]
        conv_id = ele['content'][1]["id"]
        speakers = set([uttr["speaker"] for uttr in utterances])
        t_time = [[uttr["start"], uttr["end"]]  for uttr in utterances if uttr["speaker"] == 'A']
        t_time.sort()
        
        t_start, t_end = t_time[0][0], t_time[-1][-1]
        audio_dict = {
            "audio":self.audio_root / f"{conv_id}_a.wav", "audio_start": t_start, "audio_end": t_end
        }
        
        return audio_dict
        
    def getitem(self, index: int) -> dict:
        record   = self.load_record(index)
        stream = record[1]["content"][0]["text_stream"]
        
        audio_dict = self._resolve_audio(record[0])

        # ── build streaming multi-turn conversation ───────────────────────────
        conversation, chunks_list = build_streaming_conversation(
            audio_dict["A"], audio_dict["B"], stream, self.query,
            chunk_secs=self.chunk_secs, sr=self.sr,
        )
          
        # ── tokenize ─────────────────────────────────────────────────────────
        
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=text,
            audio=[chunks_list.numpy()],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=False,
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
        inputs["text"] = text
        
        return inputs

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, index: int) -> dict:
        return self.getitem(index)
        # max_tries = 100
        # for _ in range(max_tries):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         logger.warning(f"Failed {_}-th try to get item {index}: {e}")
        #         index = random.randint(0, self.__len__() - 1)
        #         logger.warning(f"Retrying to get item {index}")
        # raise Exception(f"Failed to get item after {max_tries} retries")
        

    def data_collator(self, batched_inputs: list[dict], **kwargs) -> dict:
        """
        Collator for single-channel ASR dataset.

        Supports batch_size > 1.
        Pads:
        - input_ids with tokenizer.pad_token_id
        - attention_mask with 0
        - labels with -100
        - audio input_values / input_features with 0
        - audio padding_mask with False
        """
        
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        # ---------------------------------------------------------
        # 1. text padding
        # ---------------------------------------------------------
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for x in batched_inputs:
            input_ids_list.append(x["input_ids"].squeeze(0))
            attention_mask_list.append(x["attention_mask"].squeeze(0))
            labels_list.append(x["labels"].squeeze(0))

        max_text_len = max(t.size(0) for t in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(
            input_ids_list, attention_mask_list, labels_list
        ):
            cur_len = input_ids.size(0)
            pad_len = max_text_len - cur_len

            padded_input_ids.append(
                torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (pad_len,),
                            fill_value=pad_token_id,
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        ),
                    ],
                    dim=0,
                )
            )

            padded_attention_mask.append(
                torch.cat(
                    [
                        attention_mask,
                        torch.zeros(
                            pad_len,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=0,
                )
            )

            padded_labels.append(
                torch.cat(
                    [
                        labels,
                        torch.full(
                            (pad_len,),
                            fill_value=-100,
                            dtype=labels.dtype,
                            device=labels.device,
                        ),
                    ],
                    dim=0,
                )
            )

        input_ids = torch.stack(padded_input_ids, dim=0)
        attention_mask = torch.stack(padded_attention_mask, dim=0)
        labels = torch.stack(padded_labels, dim=0)

        # ---------------------------------------------------------
        # 2. audio padding
        # ---------------------------------------------------------
        input_features = torch.stack([f["input_features"].squeeze(0) for f in batched_inputs], dim=0)
        feature_attention_mask = torch.stack([f["feature_attention_mask"].squeeze(0) for f in batched_inputs], dim=0)

        # ---------------------------------------------------------
        # 3. keep optional fields
        # ---------------------------------------------------------
        texts = [x.get("text", "") for x in batched_inputs]
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
            "feature_mask": feature_attention_mask,
            "texts": texts
        }

        return batch


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
    
    dir = "/n/work6/yizhang/Moris/reazonspeech/shards_train"

    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    
    ds = SingleChannelASRDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=proc,
    )
    print(f"Dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=ds.data_collator)

    for batch in tqdm.tqdm(loader):
        logging.info(f"input_ids : {batch['input_ids'].size()}")
        logging.info(f"labels: {batch['labels'].size()}")

        logging.info(f"input_features : {batch['input_features'].size()}")
        logging.info(f"texts : {batch['texts']}")
    
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
        