import re
import json
import random
from pathlib import Path
from typing import Optional
import numpy as np

import torch
from torchcodec.decoders import AudioDecoder
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer, AutoFeatureExtractor, logging
from transformers.feature_extraction_utils import BatchFeature
import sys
sys.path.append("../")

from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, 
                       DEFAULT_SAMPLE_RATE, MODAL_INDEX_MAP, DEFAULT_CONTEXT_LENGTH)

logger = logging.get_logger(__name__)

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

    if multimodal_tokens is None:
        input_ids = tokenizer(texts, add_special_tokens=False).input_ids
    else:
        if isinstance(multimodal_tokens, str):
            multimodal_tokens = [multimodal_tokens]

        token2idx = {}
        for token in multimodal_tokens:
            idx = MODAL_INDEX_MAP.get(token, None)
            if idx is None:
                raise ValueError(f"{token} not found in MODAL_INDEX_MAP")
            token2idx[token] = idx

        pattern = "(" + "|".join(map(re.escape, multimodal_tokens)) + ")"
        chunks = re.split(pattern, texts)

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
    query="",
):
    """
    Build single-channel conversation.

    Keep the original return style:
    - conversation: Qwen-style multi-turn conversation
    - chunks_list: [chunk, ...]

    audio_list item format:
    {
        "audio": {"audio": path, "audio_start": start, "audio_end": end},
        "text": target_text,
    }
    """

    conversation = []
    chunks_list = []

    for chunk_info in audio_list:
        audio_info = chunk_info["audio"]
        chunk, _, _ = read_audio(audio_info)
        chunks_list.append(chunk)

        assistant_text = chunk_info.get("text", "").strip()

        if query:
            user_content = query
        else:
            user_content = ""

        # 单通道：只放一个 audio placeholder。
        user_content += "<audio>"

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



class SingleChannelASRDataset(Dataset):
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
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        sample_rate:  int   = DEFAULT_SAMPLE_RATE,
        query:        Optional[str] = None,
    ):
        super().__init__()

        self.tokenizer      = tokenizer
        self.processor      = processor
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
                # 兼容两种 jsonl：
                # 1) 旧格式：最后一行是 seek index list
                # 2) 新格式：每一行就是一个 [user, assistant] 样本
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
        
    def _resolve_audio(self, ele: list) -> list[dict]:
        """
        Resolve one jsonl record.

        Current jsonl line format:
        [
          {"role": "user", "content": [{"type": "audio", "audio": path, "start": s, "end": e}]},
          {"role": "assistant", "content": [{"type": "text", "text": text}]}
        ]
        """
        user_msg = ele[0]
        assistant_msg = ele[1]

        audio_ele = None
        for c in user_msg["content"]:
            if c.get("type") == "audio":
                audio_ele = c
                break
        if audio_ele is None:
            raise ValueError(f"No audio content found in user message: {user_msg}")

        text = ""
        for c in assistant_msg["content"]:
            if c.get("type") == "text":
                text = c.get("text", "")
                break

        audio_dict = {
            "audio": {
                "audio": audio_ele["audio"],
                "audio_start": float(audio_ele.get("start", audio_ele.get("audio_start", 0.0))),
                "audio_end": float(audio_ele.get("end", audio_ele.get("audio_end", 0.0))),
            },
            "text": text,
        }

        return [audio_dict]

    # ── Core item builder ────────────────────────────────────────────────────

    def getitem(self, index: int) -> dict:
        record   = self.load_record(index)
         
        audio_list = self._resolve_audio(record)
        # ── build streaming multi-turn conversation ───────────────────────────
        conversation, chunks_list = build_conversation(
            audio_list, self.query
        )
# ── inject real audio tensors into user content placeholders ─────────
        audio_inputs: list[torch.Tensor] = []
        chunk_idx = 0
        for msg in conversation:
            if msg["role"] != "user":
                continue
            chunk = chunks_list[chunk_idx]
            chunk_idx += 1
            pattern = "(" + "|".join(map(re.escape, [el for el in MODAL_INDEX_MAP.keys()])) + ")"
            chunks = re.split(pattern, msg["content"])
            
            for el in chunks:
                if el == "<audio>":
                    audio_inputs.append(chunk)
                                                
        # ── tokenize ─────────────────────────────────────────────────────────
        
        texts = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        texts = texts.replace("\n<think>\n\n</think>\n", "")
        input_ids = tokenizer_multimodal_token(
            texts, tokenizer=self.tokenizer, 
            multimodal_tokens=[el for el in MODAL_INDEX_MAP.keys()], return_tensors='pt'
        )
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        audios = self.processor(
            [a.numpy() for a in audio_inputs],
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=False
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
        # return self.getitem(index)
        max_tries = 100
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
        Collator for single-channel ASR dataset.

        Supports batch_size > 1.
        Pads:
        - input_ids with tokenizer.pad_token_id
        - attention_mask with 0
        - labels with -100
        - audio input_values / input_features with 0
        - audio padding_mask with False
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

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
        audio_keys = set()
        for x in batched_inputs:
            audio_keys.update(x["audios"].keys())

        audios = {}

        if "input_values" in audio_keys:
            values_list = []

            for x in batched_inputs:
                v = x["audios"]["input_values"]

                if v.dim() == 2 and v.size(0) == 1:
                    v = v.squeeze(0)

                values_list.append(v)

            max_audio_len = max(v.size(0) for v in values_list)

            padded_values = []
            padding_masks = []

            for v in values_list:
                cur_len = v.size(0)
                pad_len = max_audio_len - cur_len

                padded_values.append(
                    torch.cat(
                        [
                            v,
                            torch.zeros(
                                pad_len,
                                dtype=v.dtype,
                                device=v.device,
                            ),
                        ],
                        dim=0,
                    )
                )

                padding_masks.append(
                    torch.cat(
                        [
                            torch.ones(
                                cur_len,
                                dtype=torch.bool,
                                device=v.device,
                            ),
                            torch.zeros(
                                pad_len,
                                dtype=torch.bool,
                                device=v.device,
                            ),
                        ],
                        dim=0,
                    )
                )

            audios["input_values"] = torch.stack(padded_values, dim=0)
            audios["padding_mask"] = torch.stack(padding_masks, dim=0)

        if "input_features" in audio_keys:
            feats_list = []

            for x in batched_inputs:
                f = x["audios"]["input_features"]

                if f.dim() >= 2 and f.size(0) == 1:
                    f = f.squeeze(0)

                feats_list.append(f)

            max_feat_len = max(f.size(0) for f in feats_list)

            padded_feats = []
            feat_padding_masks = []

            for f in feats_list:
                cur_len = f.size(0)
                pad_len = max_feat_len - cur_len

                pad_shape = list(f.shape)
                pad_shape[0] = pad_len

                padded_feats.append(
                    torch.cat(
                        [
                            f,
                            torch.zeros(
                                pad_shape,
                                dtype=f.dtype,
                                device=f.device,
                            ),
                        ],
                        dim=0,
                    )
                )

                feat_padding_masks.append(
                    torch.cat(
                        [
                            torch.ones(
                                cur_len,
                                dtype=torch.bool,
                                device=f.device,
                            ),
                            torch.zeros(
                                pad_len,
                                dtype=torch.bool,
                                device=f.device,
                            ),
                        ],
                        dim=0,
                    )
                )

            audios["input_features"] = torch.stack(padded_feats, dim=0)

            # 如果前面 input_values 没有生成 padding_mask，则用 feature 级 mask
            if "padding_mask" not in audios:
                audios["padding_mask"] = torch.stack(feat_padding_masks, dim=0)

        # ---------------------------------------------------------
        # 3. keep optional fields
        # ---------------------------------------------------------
        texts = [x.get("texts", "") for x in batched_inputs]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audios": BatchFeature(audios),
            "texts": texts,
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
    
    dir = "/n/work6/yizhang/Moris/reazonspeech/train"

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("reazon-research/japanese-zipformer-base-k2-rs35kh-bpe")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    ds = SingleChannelASRDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        tokenizer=tokenizer,
        processor=proc,
    )
    print(f"Dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8, collate_fn=ds.data_collator)

    for batch in tqdm.tqdm(loader):
        multimodal_token_nums = torch.where(sum([batch['input_ids'] == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
        logging.info(f"Multimodal tokens in input_ids: {len(multimodal_token_nums)}")
        logging.info(f"input_ids : {batch['input_ids'].size()}")
        logging.info(f"labels: {batch['labels'].size()}")

        logging.info(f"audios : {batch['audios'].input_values.size()}")
        logging.info(f"texts : {batch['texts']}")
        if len(multimodal_token_nums) != batch['audios'].input_values.size(0):
            breakpoint()
        