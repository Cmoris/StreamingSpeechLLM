import json
from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torchcodec.decoders import AudioDecoder
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, AutoFeatureExtractor
import sys
sys.path.append("../")
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_audio(ele: dict):
    audio_decoder = AudioDecoder(source=ele['audio'], sample_rate=DEFAULT_SAMPLE_RATE)
    audio_sr = audio_decoder.metadata.sample_rate
    audio_duration = audio_decoder.metadata.duration_seconds_from_header
    total_frames = int(audio_duration*audio_sr)
    audio_pts = np.linspace(1/audio_sr, audio_duration, total_frames)
    audio_start = ele.get("audio_start", None)
    audio_end = ele.get("audio_end", None)
    clip_idxs = None
    if audio_start is not None or audio_end is not None:
        audio_start = audio_pts[0] if not audio_start else audio_start
        audio_end = audio_pts[-1] if not audio_end else audio_end
        clip_idxs = ((audio_start <= audio_pts) & (audio_pts <= audio_end)).nonzero()[0]
        clip_pts = audio_pts[clip_idxs]
        total_frames = len(clip_pts)
    else:
        audio_start = 0
        audio_end = audio_duration
        
    nframes = int(total_frames/audio_sr*DEFAULT_SAMPLE_RATE)
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip_pts = audio_pts[clip_idxs]
    clip = audio_decoder.get_samples_played_in_range(start_seconds=audio_start, stop_seconds=audio_end+1/DEFAULT_SAMPLE_RATE).data
    
    return clip.squeeze(0), clip_pts, audio_sr


def get_transcript_before_time(
    stream: list[dict],
    t_end: float,
    start_from: int = 0,
) -> tuple[str, int]:
    parts: list[str] = []
    i = start_from
    current_speaker = None
    current_kind = None
    seen_speakers: set[str] = set()  # 记录本次调用出现过的 speaker

    while i < len(stream) and stream[i]["end"] < t_end:
        ev = stream[i]
        spk = ev.get("speaker")
        kind = ev.get("kind")
        
        if kind == "ts":
            if current_kind == "bc":
                parts.append(BC_TOKEN[1])
                current_kind = None
            if current_speaker is not None:
                parts.append(SPEAKER_TOKENS[current_speaker][1])
                current_speaker = None
            parts.append(TS_TOKEN)

        elif kind == "te":
            if current_kind == "bc":
                parts.append(BC_TOKEN[1])
                current_kind = None
            if current_speaker is not None:
                parts.append(SPEAKER_TOKENS[current_speaker][1])
                current_speaker = None
            parts.append(TE_TOKEN)
            
        # elif kind == "asr":
        #     if current_kind == "bc":
        #         parts.append(BC_TOKEN[1])
        #         current_kind = None
        #     if spk != current_speaker:
        #         if current_speaker is not None:
        #             parts.append(SPEAKER_TOKENS[current_speaker][1])
        #         parts.append(SPEAKER_TOKENS[spk][0])
        #         current_speaker = spk
            
        #     seen_speakers.add(spk)
            
        elif kind == "bc":
            if spk != current_speaker:
                if current_speaker is not None:
                    parts.append(SPEAKER_TOKENS[current_speaker][1])
                parts.append(SPEAKER_TOKENS[spk][0])
                current_speaker = spk
                if current_kind == "bc":
                    parts.append(BC_TOKEN[1])
            if current_kind != "bc":
                parts.append(BC_TOKEN[0])
                current_kind = "bc"
            seen_speakers.add(spk)
            parts.append(ev["token"])
        
        elif kind == "silence":
            if current_kind == "bc":
                parts.append(BC_TOKEN[1])
                current_kind = None
            if current_speaker is not None:
                parts.append(SPEAKER_TOKENS[current_speaker][1])
                current_speaker = None
            parts.append(SILENCE_TOKEN)
            
        elif kind == "pause":
            if current_kind == "bc":
                parts.append(BC_TOKEN[1])
                current_kind = None
            if current_speaker is not None:
                parts.append(SPEAKER_TOKENS[current_speaker][1])
                current_speaker = None
            parts.append(PAUSE_TOKEN)

        i += 1

    # 关闭还开着的标签
    if current_kind == "bc":
        parts.append(BC_TOKEN[1])
    if current_speaker is not None:
        parts.append(SPEAKER_TOKENS[current_speaker][1])

    result = "".join(parts)

    if "A" not in seen_speakers and "B" not in seen_speakers:
        result = ""

    return result, i


# ─────────────────────────────────────────────────────────────────────────────
# Streaming conversation builder  (core of LiveCC adaptation)
# ─────────────────────────────────────────────────────────────────────────────

def safe_chunk(wav: torch.Tensor, start: int, end: int, chunk_samples: int) -> torch.Tensor:
    chunk = wav[start:end]
    if len(chunk) < chunk_samples:
        pad = make_dummy_audio(chunk_samples - len(chunk))  
        chunk = torch.cat([chunk, pad])
    return chunk

def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    """用极小噪声代替静音，避免被 feature extractor 当成 padding 截断"""
    return torch.randn(num_samples) * noise_scale

def build_streaming_conversation(
    audio_a: dict,
    audio_b: dict,
    stream: list[dict],
    query: str,
    chunk_secs: float = DEFAULT_CHUNK_SECS,
    sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[list[dict], list[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Slice both audio channels into fixed-length chunks and pair each chunk
    with the partial transcript accumulated up to that moment.

    Returns
    -------
    conversation  : ChatML list  (user/assistant alternating)
    chunks_list   : [(chunk_a, chunk_b), ...]  parallel to the user turns
    """
    chunk_samples = int(chunk_secs * sr)
    
    wav_a, clip_pts_a, _ = read_audio(audio_a) if audio_a else (make_dummy_audio(chunk_samples), np.array([]), None)
    wav_b, clip_pts_b, _ = read_audio(audio_b) if audio_b else (make_dummy_audio(chunk_samples), np.array([]), None)

    total_samples = max(len(wav_a), len(wav_b))

    # Pad to a multiple of chunk_samples so every chunk is identical length
    remainder = total_samples % chunk_samples
    if remainder != 0:
        total_samples = total_samples + (chunk_samples - remainder)

    if len(wav_a) < total_samples:
        wav_a = torch.nn.functional.pad(wav_a, (0, total_samples - len(wav_a)))
    if len(wav_b) < total_samples:
        wav_b = torch.nn.functional.pad(wav_b, (0, total_samples - len(wav_b)))

    offset = audio_a["audio_start"] if audio_a else audio_b["audio_start"] if audio_b else 0
    conversation: list[dict] = []
    chunks_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    next_start_from = 0
    first_turn = True

    for chunk_start in range(0, total_samples, chunk_samples):
        chunk_end   = min(chunk_start + chunk_samples, total_samples)
        t_start     = chunk_start / sr
        t_end       = chunk_end   / sr

        chunk_a = safe_chunk(wav_a, chunk_start, chunk_end, chunk_samples)
        chunk_b = safe_chunk(wav_b, chunk_start, chunk_end, chunk_samples)

        phrase, next_start_from = get_transcript_before_time(
            stream, t_end+offset, start_from=next_start_from
        )
        
        # ── user turn ────────────────────────────────────────────────────────
        user_content: list[dict] = [
            {"type": "text",  "text": f"Time={t_start:.1f}-{t_end:.1f}s"},
            {"type": "audio", "audio": "__A__"},   # placeholder, resolved in __getitem__
            {"type": "audio", "audio": "__B__"},
        ]
        if first_turn:
            user_content.append({"type": "text", "text": query})
            first_turn = False

        # ── assistant turn  (" ..." = streaming not yet complete) ────────────
        if phrase == "":
            assistant_text = STREAMING_CONT
        else:
            assistant_text = phrase

        conversation.append({"role": "user",      "content": user_content})
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        chunks_list.append((chunk_a, chunk_b))

    # ── trim trailing empty assistant turns (no new tokens vs previous) ──────
    while len(conversation) >= 2:
        last_asst = conversation[-1]["content"][0]["text"]
        prev_asst = conversation[-3]["content"][0]["text"] if len(conversation) >= 4 else ""
        # both end with STREAMING_CONT and transcript part is identical → prune
        if (last_asst == prev_asst) or (last_asst == STREAMING_CONT):
            conversation = conversation[:-2]
            chunks_list  = chunks_list[:-1]
        else:
            break

    # ── finalize last assistant turn: remove " ..." suffix ───────────────────
    if conversation and conversation[-1]["role"] == "assistant":
        last_text = conversation[-1]["content"][0]["text"]
        if last_text.endswith(STREAMING_CONT):
            conversation[-1]["content"][0]["text"] = last_text[: -len(STREAMING_CONT)]

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
    max_length       : truncate token sequences to this length (-1 = off)
    query            : instruction prepended to the first user turn
    audio_path_map   : optional {conv_id: {"A": path, "B": path}} override
    """

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
        audio_root_a: str,
        audio_root_b: str,
        chunk_secs:   float = DEFAULT_CHUNK_SECS,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
    ):
        super().__init__()

        self.processor      = processor
        self.audio_root_a   = Path(audio_root_a)
        self.audio_root_b   = Path(audio_root_b)
        self.chunk_secs     = chunk_secs
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
        return self.getitem(index)

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
    
    dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/l4_conv_train"

    from transformers import Qwen2AudioProcessor
    proc = Qwen2AudioProcessor.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
    )
    audio_token_id = proc.tokenizer("<|AUDIO|>").input_ids[0]
    ds = DualChannelStreamingConvDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=proc,
        audio_root_a="/n/work6/yizhang/Moris/zoom2025/audios/A_gd",
        audio_root_b="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
        chunk_secs=DEFAULT_CHUNK_SECS,
    )
    print(f"Dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=ds.data_collator)

    for batch in tqdm.tqdm(loader):
        multimodal_token_nums = torch.where(sum([batch['input_ids'][0] == audio_token_id]))[0]
        logging.info(f"Multimodal tokens in input_ids: {len(multimodal_token_nums)}")
        logging.info(f"input_ids : {batch['input_ids'].size()}")
        logging.info(f"labels: {batch['labels'].size()}")
        logging.info(f"audios : {batch['input_features'].size()}")
        target_ids = batch["input_ids"]
        target_ids[target_ids == -100] = proc.tokenizer.pad_token_id
        refs = proc.batch_decode(target_ids, skip_special_tokens=True)
        logging.info(f"texts : {refs}")