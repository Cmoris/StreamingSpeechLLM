import json
import random
from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, AutoFeatureExtractor, logging

import sys
sys.path.append("../")
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN, QUERY,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE)
from data_utils import read_audio, safe_chunk, make_dummy_audio, pad_audio_to_min_len

logger = logging.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    audio_a: dict | None,
    audio_b: dict | None,
    stream: list[dict],
    query: str,
    chunk_secs: float = DEFAULT_CHUNK_SECS,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[list[dict], list[tuple[torch.Tensor, torch.Tensor]]]:

    chunk_samples = int(chunk_secs * sr)

    wav_a, _, _ = read_audio(audio_a) if audio_a else (
        make_dummy_audio(chunk_samples), np.array([]), None
    )
    wav_b, _, _ = read_audio(audio_b) if audio_b else (
        make_dummy_audio(chunk_samples), np.array([]), None
    )

    total_samples = max(len(wav_a), len(wav_b))

    remainder = total_samples % chunk_samples
    if remainder != 0:
        total_samples += chunk_samples - remainder

    if len(wav_a) < total_samples:
        wav_a = torch.nn.functional.pad(wav_a, (0, total_samples - len(wav_a)))
    if len(wav_b) < total_samples:
        wav_b = torch.nn.functional.pad(wav_b, (0, total_samples - len(wav_b)))

    # stream 里面的时间是全局时间，因此 chunk 的相对时间需要加 offset
    if audio_a is not None:
        offset = audio_a.get("audio_start", 0.0)
    elif audio_b is not None:
        offset = audio_b.get("audio_start", 0.0)
    else:
        offset = 0.0

    conversation: list[dict] = []
    chunks_list: list[tuple[torch.Tensor, torch.Tensor]] = []

    next_start_from = 0
    first_turn = True

    for chunk_start in range(0, total_samples, chunk_samples):
        chunk_end = chunk_start + chunk_samples

        t_start = chunk_start / sr
        t_end = chunk_end / sr

        abs_t_end = offset + t_end

        chunk_a = safe_chunk(wav_a, chunk_start, chunk_end, chunk_samples)
        chunk_b = safe_chunk(wav_b, chunk_start, chunk_end, chunk_samples)

        phrase, next_start_from = get_transcript_before_time(
            stream=stream,
            t_end=abs_t_end,
            start_from=next_start_from,
        )

        user_content = [
            {"type": "text", "text": "Speaker A"},
            {"type": "audio", "audio": "__A__"},
            {"type": "text", "text": "Speaker B"},
            {"type": "audio", "audio": "__B__"},
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

        chunks_list.append((chunk_a, chunk_b))

    while len(conversation) >= 2:
        last_text = conversation[-1]["content"][0]["text"]
        if last_text == STREAMING_CONT:
            conversation = conversation[:-2]
            chunks_list = chunks_list[:-1]
        else:
            break

    return conversation, chunks_list


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DualChannelStreamingConvDataset(Dataset):
    """
    Streaming dual-channel ASR dataset for Qwen2-Audio fine-tuning,
    following the LiveCC incremental-chunk paradigm.

    Parameters
    ----------
    annotation_paths : list of .jsonl files (last line = seek-index array)
                       OR list of .json files (one record per file, no seek)
    processor        : Qwen2AudioProcessor (already loaded + special tokens added)
    audio_root_a/b   : root dirs for speaker A / B wav files
    chunk_secs       : duration of each streaming audio chunk (seconds)
    sample_rate      : target sampling rate
    query            : instruction prepended to the first user turn
    """

    SYSTEM_PROMPT = ""

    QUERY = ""

    def __init__(
        self,
        annotation_paths: list[str],
        processor: Optional[AutoProcessor],
        audio_root_a: str,
        audio_root_b: str,
        chunk_secs:   float = DEFAULT_CHUNK_SECS,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
        eval:         bool = False
    ):
        super().__init__()

        self.processor      = processor
        self.audio_root_a   = Path(audio_root_a)
        self.audio_root_b   = Path(audio_root_b)
        self.chunk_secs     = chunk_secs
        self.sr             = sample_rate
        self.query          = query or self.QUERY
        self.eval           = eval
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

    def _resolve_audio(self, ele: dict) -> tuple[Path, Path]:
        utterances = ele['content'][0]["utterances"]
        conv_id = ele['content'][1]["id"]
        speakers = set([uttr["speaker"] for uttr in utterances])
        t_A_time = [[uttr["start"], uttr["end"]]  for uttr in utterances if uttr["speaker"] == 'A']
        t_B_time = [[uttr["start"], uttr["end"]]  for uttr in utterances if uttr["speaker"] == 'B']
        t_A_time.sort()
        t_B_time.sort()
        
        if 'A' in speakers and 'B' in speakers:
            t_A_start, t_A_end = t_A_time[0][0], t_A_time[-1][-1]
            t_B_start, t_B_end = t_B_time[0][0], t_B_time[-1][-1]
            t_start = min(t_A_start, t_B_start)
            t_end = max(t_A_end, t_B_end)
            audio_dict = {
                "A": {"audio":self.audio_root_a / f"{conv_id}_a.wav", "audio_start": t_start, "audio_end": t_end},
                "B": {"audio":self.audio_root_b / f"{conv_id}_b.wav", "audio_start": t_start, "audio_end": t_end}
            }
        elif 'A' in speakers:
            t_A_start, t_A_end = t_A_time[0][0], t_A_time[-1][-1]
            t_start, t_end = t_A_start, t_A_end
            audio_dict = {
                "A": {"audio":self.audio_root_a / f"{conv_id}_a.wav", "audio_start": t_start, "audio_end": t_end},
                "B": None
            }
        else:
            t_B_start, t_B_end = t_B_time[0][0], t_B_time[-1][-1]
            t_start, t_end = t_B_start, t_B_end
            audio_dict = {
                "A": None,
                "B": {"audio":self.audio_root_b / f"{conv_id}_b.wav", "audio_start": t_start, "audio_end": t_end}
            }
        return audio_dict

    # ── Core item builder ────────────────────────────────────────────────────

    def getitem(self, index: int) -> dict:
        record   = self.load_record(index)
        stream = record[1]["content"][0]["text_stream"]
        
        audio_dict = self._resolve_audio(record[0])

        # ── build streaming multi-turn conversation ───────────────────────────
        conversation, chunks_list = build_streaming_conversation(
            audio_dict["A"], audio_dict["B"], stream, self.query,
            chunk_secs=self.chunk_secs, sr=self.sr,
        )
# ── inject real audio tensors into user content placeholders ─────────
        audio_inputs: list[torch.Tensor] = []
        chunk_idx = 0
        for msg in conversation:
            if msg["role"] != "user":
                continue
            chunk_a, chunk_b = chunks_list[chunk_idx]
            chunk_idx += 1
            for el in msg["content"]:
                if el["type"] == "audio":
                    if el["audio"] == "__A__":
                        el["audio"] = chunk_a.numpy()
                        audio_inputs.append(chunk_a)
                    elif el["audio"] == "__B__":
                        el["audio"] = chunk_b.numpy()
                        audio_inputs.append(chunk_b)
                                                
        # # ── prepend system turn ───────────────────────────────────────────────
        # full_conv = [{"role": "system", "content": self.SYSTEM_PROMPT}] + conversation

        # ── tokenize ─────────────────────────────────────────────────────────
        
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True if self.eval else False
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: add special tokens before constructing dataset
# ─────────────────────────────────────────────────────────────────────────────

def add_special_tokens(processor: AutoProcessor, model) -> None:
    n = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [TS_TOKEN, TE_TOKEN]}
    )
    if n:
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info(f"Added {n} special token(s): {[TS_TOKEN, TE_TOKEN]}")

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
    
    dir = "/n/work6/yizhang/Moris/zoom2025/pretrain_labels/l8_conv_train_with_backchannel"

    from transformers import Qwen2AudioProcessor
    proc = Qwen2AudioProcessor.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
    )
    audio_token_id = proc.tokenizer("<|AUDIO|>").input_ids[0]
    ds = DualChannelStreamingConvDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=proc,
        audio_root_a="/n/work6/yizhang/Moris/zoom2025/audios/A_all",
        audio_root_b="/n/work6/yizhang/Moris/zoom2025/audios/B_all",
        chunk_secs=DEFAULT_CHUNK_SECS,
        query=QUERY
    )
    print(f"Dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.data_collator)

    for batch in tqdm.tqdm(loader):
        multimodal_token_nums = torch.where(sum([batch['input_ids'][0] == audio_token_id]))[0]
        # logging.info(f"Multimodal tokens in input_ids: {len(multimodal_token_nums)}")
        # logging.info(f"input_ids : {batch['input_ids'].size()}")
        # logging.info(f"labels: {batch['labels'].size()}")
        # logging.info(f"audios : {batch['input_features'].size()}")
        target_ids = batch["input_ids"]
        target_ids[target_ids == -100] = proc.tokenizer.pad_token_id
        refs = proc.batch_decode(target_ids, skip_special_tokens=True)
        logging.info(f"texts : {refs}")