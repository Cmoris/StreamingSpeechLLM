import re
import json
import random
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
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, 
                       DEFAULT_SAMPLE_RATE, MODAL_INDEX_MAP, DEFAULT_CONTEXT_LENGTH)


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

def tokenizer_multimodal_token(
    texts,
    tokenizer,
    multimodal_tokens=None,  
    return_tensors=None
):
    """
    Tokenize text and multimodal tags into input_ids.

    Args:
        texts (str): Text prompt (with multimodal tags)
        tokenizer: Huggingface tokenizer
        multimodal_token (str or List[str]): multimodal tokens
        return_tensors (str): 'pt' or None
    """

    # 1. 如果没有多模态token，直接走原逻辑
    if multimodal_tokens is None:
        input_ids = tokenizer(texts, add_special_tokens=False).input_ids
    else:
        # 2. 统一成 list
        if isinstance(multimodal_tokens, str):
            multimodal_tokens = [multimodal_tokens]

        # 3. 建立 token -> index 映射
        token2idx = {}
        for token in multimodal_tokens:
            idx = MODAL_INDEX_MAP.get(token, None)
            if idx is None:
                raise ValueError(f"{token} not found in MODAL_INDEX_MAP")
            token2idx[token] = idx

        # 4. 用正则 split（保留分隔符）
        pattern = "(" + "|".join(map(re.escape, multimodal_tokens)) + ")"
        chunks = re.split(pattern, texts)

        # 5. 逐块处理
        input_ids = []
        for chunk in chunks:
            if chunk in token2idx:
                # multimodal token
                input_ids.append(token2idx[chunk])
            else:
                # normal text
                if chunk:
                    ids = tokenizer(chunk, add_special_tokens=False).input_ids
                    input_ids.extend(ids)

    # 6. tensor处理
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return input_ids

def safe_chunk(wav: torch.Tensor, start: int, end: int, chunk_samples: int) -> torch.Tensor:
    chunk = wav[start:end]
    if len(chunk) < chunk_samples:
        pad = make_dummy_audio(chunk_samples - len(chunk))  
        chunk = torch.cat([chunk, pad])
    return chunk

def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    """用极小噪声代替静音，避免被 feature extractor 当成 padding 截断"""
    return torch.randn(num_samples) * noise_scale


def build_conversation(
    audio_list,
    utterances,
    query="",
    sr=16000,
):
    """
    Build streaming dual-channel conversation.

    Returns
    -------
    conversation : list[dict]
        Qwen-style multi-turn conversation
    chunks_list : list[tuple[Tensor, Tensor]]
        [(chunk_a, chunk_b), ...]
    """

    conversation = []
    chunks_list = []

    for chunk_info in audio_list:
        
        # ---------------------------------------------------------
        # audio meta
        # ---------------------------------------------------------
        a_info = chunk_info["A"]
        b_info = chunk_info["B"]

        if a_info is not None:
            start_time = a_info["audio_start"]
            end_time = a_info["audio_end"]
        else:
            start_time = b_info["audio_start"]
            end_time = b_info["audio_end"]

        # ---------------------------------------------------------
        # load audio
        # ---------------------------------------------------------
        
        if a_info is not None:
            chunk_a, _, _ = read_audio(a_info)
        else:
            length = int((end_time - start_time) * sr)
            chunk_a = make_dummy_audio(length)

        if b_info is not None:
            chunk_b, _, _ = read_audio(b_info)
        else:
            length = int((end_time - start_time) * sr)
            chunk_b = make_dummy_audio(length)

        chunks_list.append((chunk_a, chunk_b))

        # ---------------------------------------------------------
        # collect utterances inside this chunk
        # ---------------------------------------------------------
        cur_uttrs = []

        for u in utterances:

            # overlap with chunk
            if u["end"] < start_time:
                continue

            if u["start"] > end_time:
                continue

            speaker = u["speaker"]
            text = u["text"].strip()

            if len(text) == 0:
                continue

            # optional special tags
            suffix = ""
            if u.get("is_turn_taking", False):
                suffix += TS_TOKEN

            if u.get("is_back_channel", False):
                suffix += BC_TOKEN

            if u.get("is_pause", False):
                suffix += PAUSE_TOKEN

            if u.get("is_silence", False):
                suffix += SILENCE_TOKEN

            if speaker == "A":
                spk_tag = "speaker_A"
            else:
                spk_tag = "speaker_B"

            cur_uttrs.append(
                f"<{spk_tag}>{text}</{spk_tag}>{suffix}"
            )

        assistant_text = "".join(cur_uttrs)

        # ---------------------------------------------------------
        # build one streaming turn
        # ---------------------------------------------------------
        
        if query:
            user_content = query
        else:
            user_content = ""
        
        user_content += "Speaker A: <A>, Speaker B: <B>"

        conversation.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        conversation.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

    return conversation, chunks_list



class DualChannelConvDataset(Dataset):
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
        tokenizer: Optional[AutoTokenizer],
        audio_root_a: str,
        audio_root_b: str,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
    ):
        super().__init__()

        self.tokenizer      = tokenizer
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
        ) = tokenizer("<|im_start|>assistant\n<|im_end|>").input_ids

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

        for i in range(0, len(utterances)-self.context_length+1):
            uttr = utterances[i:i+self.context_length]
            t_start = float(uttr[0]["start"])
            t_end = float(uttr[-1]["end"])

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
                "utterances": uttr,
            }

            audio_list.append(audio_dict)

        return audio_list

    # ── Core item builder ────────────────────────────────────────────────────

    def getitem(self, index: int) -> dict:
        record   = self.load_record(index)
        
        utterances = record[0]['content'][0]["utterances"]
        
        audio_list = self._resolve_audio(record[0])
        
        # ── build streaming multi-turn conversation ───────────────────────────
        conversation, chunks_list = build_conversation(
            audio_list, utterances, self.query, sr=self.sr,
        )
# ── inject real audio tensors into user content placeholders ─────────
        audio_inputs: list[torch.Tensor] = []
        chunk_idx = 0
        for msg in conversation:
            if msg["role"] != "user":
                continue
            chunk_a, chunk_b = chunks_list[chunk_idx]
            chunk_idx += 1
            pattern = "(" + "|".join(map(re.escape, [el for el in MODAL_INDEX_MAP.keys()])) + ")"
            chunks = re.split(pattern, msg["content"])
            
            for el in chunks:
                if el == "<A>":
                    audio_inputs.append(chunk_a)
                elif el == "<B>":
                    audio_inputs.append(chunk_b)
                                                
        # ── tokenize ─────────────────────────────────────────────────────────
        
        texts = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        
        input_ids = tokenizer_multimodal_token(
            texts, tokenizer=self.tokenizer, 
            multimodal_tokens=[el for el in MODAL_INDEX_MAP.keys()], return_tensors='pt'
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        audios = self.processor(
            [a.numpy() for a in audio_inputs],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=True
        )
        audios["padding_mask"] = getattr(audios, "input_values", getattr(audios, "input_features", None)) != 0
        
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
    
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audios=audios,
            texts=texts
        )
    
        return batch

    def __len__(self) -> int:
        return len(self.handles)

    def __getitem__(self, index: int) -> dict:
        return self.getitem(index)
        # max_tries = 10
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
    
    dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_train_with_backchannel"

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("reazon-research/japanese-zipformer-base-k2-rs35kh-bpe")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    ds = DualChannelConvDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        tokenizer=tokenizer,
        processor=proc,
        audio_root_a="/n/work6/yizhang/Moris/zoom2025/audios/A_gd",
        audio_root_b="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
    )
    print(f"Dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=ds.data_collator)

    for batch in tqdm.tqdm(loader):
        multimodal_token_nums = torch.where(sum([batch['input_ids'][0] == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
        logging.info(f"Multimodal tokens in input_ids: {len(multimodal_token_nums)}")
        logging.info(f"input_ids : {batch['input_ids'].size()}")
        logging.info(f"labels: {batch['labels'].size()}")

        logging.info(f"audios : {batch['audios'].input_values.size()}")
        logging.info(f"texts : {batch['texts']}")
        pass