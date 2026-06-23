import os
import json
import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from peft import PeftModel
from transformers import (
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
    AutoProcessor,
    LogitsProcessor,
    logging,
    HfArgumentParser
)

from torchcodec.decoders import AudioDecoder

from constants import DEFAULT_SAMPLE_RATE, DEFAULT_CHUNK_SECS, DEFAULT_CONTEXT_LENGTH, TE_TOKEN, TS_TOKEN, SPEAKER_TOKENS, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN


logger = logging.get_logger(__name__)


@dataclass
class DataArguments:
    annotation_dir: str = field(default="")
    audio_root_a: str = field(default="")
    audio_root_b: str = field(default="")
    initial_chunk_sec: float = field(default=0)
    streaming_chunk_sec: float = field(default=DEFAULT_CHUNK_SECS)
    sample_rate: int = field(default=DEFAULT_SAMPLE_RATE)
    query: str = field(default=None)
    output_dir: str = field(default="./results")

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(default='Qwen/Qwen2-Audio-7B-Instruct')
    model_path: str = field(default="/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR_TS/qwen2audio_finetunel2_chunk1s_lora16/checkpoint-3000")


class ThresholdLogitsProcessor(LogitsProcessor):
    def __init__(self, token_id: int, base_threshold: float, step: float):
        self.token_id = token_id
        self.base_threshold = base_threshold
        self.step = step
        self.count = 0

    def __call__(self, input_ids, scores):
        threshold = self.base_threshold + self.step * self.count
        low_confidence = (
            torch.softmax(scores, dim=-1)[:, self.token_id] <= threshold
        )

        if low_confidence.any():
            scores[low_confidence, self.token_id] = -float("inf")

        self.count += 1
        return scores

def get_past_len(past_key_values):
    if past_key_values is None:
        return 0

    # transformers 新版 Cache 对象
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()

    # legacy tuple: layer -> (key, value)
    return past_key_values[0][0].shape[-2]


class LiveCCAudioInfer:

    AUDIO_END = object()

    sample_rate = 16000

    def __init__(
        self,
        initial_chunk_sec: float,
        streaming_chunk_sec: float,
        pretrained_model_name_or_path: str,
        model_path: str,
        device: str = None,
    ):

        self.initial_chunk_sec = initial_chunk_sec
        self.streaming_chunk_sec = streaming_chunk_sec

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            dtype=torch.float16,
            device_map=device,
        )
        
        if model_path:
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
            )

            self.model = model.merge_and_unload()
        else:
            self.model = base_model
        
        self.model.eval()

        self.processor = Qwen2AudioProcessor.from_pretrained(
            pretrained_model_name_or_path,
        )
        
        self.processor.tokenizer.add_tokens([TE_TOKEN, TS_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN, SPEAKER_TOKENS['A'][0], SPEAKER_TOKENS['A'][1], SPEAKER_TOKENS['B'][0], SPEAKER_TOKENS['B'][1]], special_tokens=False)

        self.streaming_eos_token_id = (
            self.processor.tokenizer(" ...").input_ids[-1]
        )

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "live audio cc"},
            ],
        }

        texts = self.processor.apply_chat_template(
            [message],
            tokenize=False,
        )

        self.system_prompt_offset = texts.index("<|im_start|>user")

        
    def load_audio(self, audio_path):

        decoder = AudioDecoder(audio_path)

        waveform = decoder.get_all_samples().data

        # [C, T] -> mono
        if waveform.dim() == 2:
            waveform = waveform.mean(0)

        sample_rate = decoder.metadata.sample_rate

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate,
            )

        return waveform

    @torch.inference_mode()
    def live_cc(
        self,
        message: str,
        state: dict,
        default_query: str = "",
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        streaming_eos_base_threshold: float = None,
        streaming_eos_threshold_step: float = None,
        max_new_tokens: int = 16,
    ):

        audio_path = state.get("audio_path", None)

        if audio_path is None:
            return

        waveform = self.load_audio(audio_path)

        current_sec = state.get("audio_timestamp", 0.0)
        last_sec = state.get("last_timestamp", 0.0)

        total_sec = waveform.shape[-1] / self.sample_rate

        if last_sec >= total_sec:
            state["audio_end"] = True
            return

        initialized = last_sec > 0

        if not initialized:
            current_sec = max(
                current_sec,
                self.initial_chunk_sec,
            )

        if current_sec <= last_sec:
            return

        if not initialized:
            chunk_size = self.initial_chunk_sec
        else:
            chunk_size = self.streaming_chunk_sec

        chunks = []

        start = last_sec

        while start < current_sec:

            end = min(start + chunk_size, current_sec)

            s = int(start * self.sample_rate)
            e = int(end * self.sample_rate)

            audio_chunk = waveform[s:e]

            if audio_chunk.numel() == 0:
                break

            chunks.append((audio_chunk, start, end))

            start = end

            chunk_size = self.streaming_chunk_sec

        for audio_chunk, start_sec, end_sec in chunks:

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Time={start_sec:.1f}-{end_sec:.1f}s",
                        },
                        {
                            "type": "audio",
                            "audio": audio_chunk.numpy(),
                        },
                    ],
                }
            ]

            if not message and not state.get("message", None):
                message = default_query

            if message and state.get("message", None) != message:
                conversation[0]["content"].append(
                    {
                        "type": "text",
                        "text": message,
                    }
                )

                state["message"] = message

            texts = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=texts,
                audios=[audio_chunk.numpy()],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )

            inputs = inputs.to(self.model.device)


            if streaming_eos_base_threshold is not None:
                logits_processor = [
                    ThresholdLogitsProcessor(
                        self.streaming_eos_token_id,
                        streaming_eos_base_threshold,
                        streaming_eos_threshold_step,
                    )
                ]
            else:
                logits_processor = None

            outputs = self.model.generate(
                **inputs,
                past_key_values=state.get(
                    "past_key_values",
                    None,
                ),
                return_dict_in_generate=True,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.model.config.eos_token_id,
            )

            state["past_key_values"] = outputs.past_key_values
            state["last_timestamp"] = end_sec

            response = self.processor.decode(
                outputs.sequences[
                    0,
                    inputs.input_ids.size(1):,
                ],
                skip_special_tokens=True,
            )

            yield (
                (start_sec, end_sec),
                response,
                state,
            )
    @torch.inference_mode()
    def live_cc_once_for_evaluation(
        self,
        query: str,
        audio_A: Union[str, torch.Tensor],
        audio_B: Union[str, torch.Tensor],
        audio_start: float = 0,
        audio_end: float = None,
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.05,
    ):

        # 1. load audio
        if type(audio_A) == str:
            waveform_A = self.load_audio(audio_A)
        else:
            waveform_A = audio_A
        if type(audio_B) == str:
            waveform_B = self.load_audio(audio_B)
        else:
            waveform_B = audio_B

        total_sec = max(waveform_A.shape[-1], waveform_B.shape[-1]) / self.sample_rate

        if audio_end is None:
            audio_end = total_sec

        audio_end = min(audio_end, total_sec)

        # 2. crop audio
        s = int(audio_start * self.sample_rate)
        e = int(audio_end * self.sample_rate)
        
        waveform_A = waveform_A[s:e]
        waveform_B = waveform_B[s:e]

        # 3. split chunks
        interleave_chunks_A = []
        interleave_chunks_B = []

        initial_samples = int(
            self.initial_chunk_sec * self.sample_rate
        )

        streaming_samples = int(
            self.streaming_chunk_sec * self.sample_rate
        )

        # 3.1 initial chunk
        initial_chunk_A = waveform_A[:initial_samples]
        initial_chunk_B = waveform_B[:initial_samples]

        if initial_chunk_A.numel() > 0:
            interleave_chunks_A.append(initial_chunk_A)
        if initial_chunk_B.numel() > 0:
            interleave_chunks_B.append(initial_chunk_B)

        # 3.2 streaming chunks
        remain_A = waveform_A[initial_samples:]
        remain_B = waveform_B[initial_samples:]

        if remain_A.numel() > 0 or remain_B.numel() > 0:
            for i in range(
                0,
                max(remain_A.shape[-1], remain_B.shape[-1]),
                streaming_samples,
            ):

                chunk_A = remain_A[
                    i:i + streaming_samples
                ]
                chunk_B = remain_B[
                    i:i + streaming_samples
                ]

                if chunk_A.numel() > 0:
                    interleave_chunks_A.append(chunk_A)
                if chunk_B.numel() > 0:
                    interleave_chunks_B.append(chunk_B)

        # 4. streaming inference
        past_key_values = None

        responses = []

        current_time = audio_start

        for i, (audio_chunk_A, audio_chunk_B) in enumerate(zip(interleave_chunks_A, interleave_chunks_B)):
            chunk_sec = (
                audio_chunk_A.shape[-1] / self.sample_rate
            )

            start_timestamp = current_time
            stop_timestamp = current_time + chunk_sec

            current_time = stop_timestamp

            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Time={start_timestamp:.1f}-"
                            f"{stop_timestamp:.1f}s"
                        ),
                    },
                    {
                        "type": "audio",
                        "audio": audio_chunk_A.numpy(),
                    },
                    {
                        "type": "audio",
                        "audio": audio_chunk_B.numpy(),
                    }
                ],
            }

            if not past_key_values:
                message["content"].append(
                    {
                        "type": "text",
                        "text": query,
                    }
                )

            texts = self.processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
            )

            if past_key_values:
                texts = (
                    "<|im_end|>\n"
                    + texts[self.system_prompt_offset:]
                )

            inputs = self.processor(
                text=texts,
                audio=[audio_chunk_A.numpy(), audio_chunk_B.numpy()],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            # audio_token_id = self.processor.tokenizer("<|AUDIO|>").input_ids[0]
            # print((inputs["attention_mask"] == 1))
            # print((inputs["input_ids"] != audio_token_id))
            inputs = inputs.to(self.model.device)
            
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
                attention_mask = torch.cat(
                    [past_attention_mask, cur_attention_mask],
                    dim=1,
                )
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

            outputs = self.model.generate(
                **model_inputs,
                return_dict_in_generate=True,
                repetition_penalty=repetition_penalty,
            )

            past_key_values = outputs.past_key_values
            
            next_token_logits = outputs.logits[:, -1, :]
            
            generated_ids = []

            eos_ids = set()
            if tokenizer.eos_token_id is not None:
                eos_ids.add(tokenizer.eos_token_id)

            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
                eos_ids.add(im_end_id)
            
            for _ in range(max_new_tokens):
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                token_id = next_token.item()
                generated_ids.append(token_id)

                # 把刚生成的 token 也喂回模型，更新 KV cache。
                # 即使 token 是 <|im_end|>，也最好写入 cache，
                # 这样下一轮 user turn 接在完整 assistant turn 后面。
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

                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                if token_id in eos_ids:
                    break

                next_token_logits = outputs.logits[:, -1, :]

            response = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            
            # response = self.processor.decode(
            #     outputs.sequences[
            #         0,
            #         inputs.input_ids.size(1):,
            #     ],
            #     skip_special_tokens=True,
            # )

            responses.append(
                [
                    start_timestamp,
                    stop_timestamp,
                    response,
                ]
            )

        return responses

def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()
    
    
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

def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    """用极小噪声代替静音，避免被 feature extractor 当成 padding 截断"""
    return torch.randn(num_samples) * noise_scale

class AudioDataset(Dataset):
    def __init__(self, 
                 annotation_paths: List[str],
                 audio_root_a: str,
                 audio_root_b: str,
                 chunk_secs: int,
                 sample_rate: int = 16000):
        super().__init__()

        self.audio_root_a   = Path(audio_root_a)
        self.audio_root_b   = Path(audio_root_b)
        self.sr             = sample_rate
        self.chunk_secs      = chunk_secs

        # ── build seek-based handle list (same as LiveCC) ────────────────────
        self.handles: list[tuple[str, int]] = []
        for ap in annotation_paths:
            ap = str(ap)
            if ap.endswith(".jsonl"):
                # last line stores seek indices
                seeks = json.loads(readlastline(ap))
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
        
        chunk_samples = int(self.chunk_secs * self.sr)
        
        wav_a, clip_pts_a, _ = read_audio(audio_dict['A']) if audio_dict['A'] else (make_dummy_audio(chunk_samples), np.array([]), None)
        wav_b, clip_pts_b, _ = read_audio(audio_dict['B']) if audio_dict['B'] else (make_dummy_audio(chunk_samples), np.array([]), None)
        
        total_samples = max(len(wav_a), len(wav_b))
        
        remainder = total_samples % chunk_samples
        if remainder != 0:
            total_samples = total_samples + (chunk_samples - remainder)

        if len(wav_a) < total_samples:
            wav_a = torch.nn.functional.pad(wav_a, (0, total_samples - len(wav_a)))
        if len(wav_b) < total_samples:
            wav_b = torch.nn.functional.pad(wav_b, (0, total_samples - len(wav_b)))
        
        duration = total_samples / self.sr
        
        return wav_a, wav_b, duration
    
    def __getitem__(self, index):
        max_tries = 10
        for _ in range(max_tries):
            try:
                audio_a, audio_b, duration = self.getitem(index)
                return dict(
                    audio_A=audio_a,
                    audio_B=audio_b,
                    start=0,
                    end=duration
                )
            except Exception as e:
                logger.warning(f"Failed {_}-th try to get item {index}: {e}")
                index = random.randint(0, self.__len__() - 1)
                logger.warning(f"Retrying to get item {index}")
        raise Exception(f"Failed to get item after {max_tries} retries")

        
        
    def __len__(self):
        return len(self.handles)
    
    def data_collator(self, batched_inputs: list[dict], **kwargs) -> dict:
        """
        LiveCC-style collator: batch_size=1 only (multi-turn audio is huge).
        For batch_size>1 override this with a proper padded collator.
        """
        assert len(batched_inputs) == 1, \
            "Use batch_size=1 for streaming audio; override data_collator for larger batches."
        return batched_inputs[0]

def main():
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    output_dir = Path(data_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    infer = LiveCCAudioInfer(
        initial_chunk_sec=data_args.initial_chunk_sec,
        streaming_chunk_sec=data_args.streaming_chunk_sec,
        pretrained_model_name_or_path=model_args.pretrained_model_name_or_path,
        model_path=model_args.model_path,
        device=device
    )
    annotation_paths = [str(path) for path in Path(data_args.annotation_dir).glob("*.jsonl")]
    dataset = AudioDataset(
        annotation_paths=annotation_paths,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        chunk_secs=data_args.streaming_chunk_sec
    )
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=8,
        pin_memory=False,
        collate_fn=dataset.data_collator,
    )
    output_path = output_dir / f"Chunk{data_args.streaming_chunk_sec}_{sampler.rank}_results.jsonl"

    with open(output_path, "w") as f:
        for batch in tqdm(dataloader):
            
            responses = infer.live_cc_once_for_evaluation(
                query="",
                audio_A=batch["audio_A"],
                audio_B=batch["audio_B"],
                audio_start=batch["start"],
                audio_end=batch["end"]
            )
            # print(responses)
            f.write(json.dumps(responses, ensure_ascii=False) + "\n")
            
        
if __name__ == "__main__":
    main()