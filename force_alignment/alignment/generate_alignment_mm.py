import os
import math
import json
import argparse
import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import nltk
from nltk.data import load as nltk_load
from data import TranscriptDataset, TranscriptDatasetForTXT, DataCollatorForTranscripts
from data.schema import SingleSegment, SingleAlignedSegment, SingleWordSegment, AlignedTranscriptionResult, SegmentData
from utils.log_utils import get_logger
from utils.align_utils import load_audio, interpolate_nans, SAMPLE_RATE, PUNKT_LANGUAGES

logger = get_logger(__name__)

ERROR_TOKEN = ['。', '？', '！', '，', '、', '；', '：']

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "reazon-research/japanese-wav2vec2-large-rs35kh",
}


class Alignment:
    def __init__(
        self,
        annotation_dir: str,
        target_sr: int = 16000,
        batch_size: int = 4,
        language_code: str = "en",
        device: str = "cuda",
        model_name_or_path: Optional[str] = None,
        model_dir: str = None,
        output_dir: str = "./outputs",
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (local_rank == 0)

        self.model, self.metadata = self.load_align_model(
            language_code, device, model_name_or_path, model_dir
        )

        transcript_dataset = TranscriptDataset(
            annotation_dir=annotation_dir, target_sr=target_sr
        )

        sampler = DistributedSampler(
            transcript_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False,
            drop_last=False,
        )

        self.transcript_loader = torch.utils.data.DataLoader(
            transcript_dataset,
            batch_size=batch_size,
            sampler=sampler,          
            num_workers=4,
            collate_fn=DataCollatorForTranscripts(),
            pin_memory=False,
        )

    def load_align_model(
        self,
        language_code: str,
        device: str,
        model_name_or_path: Optional[str] = None,
        model_dir=None,
    ):
        if model_name_or_path is None:
            if language_code in DEFAULT_ALIGN_MODELS_TORCH:
                model_name_or_path = DEFAULT_ALIGN_MODELS_TORCH[language_code]
            elif language_code in DEFAULT_ALIGN_MODELS_HF:
                model_name_or_path = DEFAULT_ALIGN_MODELS_HF[language_code]
            else:
                logger.error(
                    f"No default alignment model for language: {language_code}."
                )
                raise ValueError(f"No default align-model for language: {language_code}")

        if model_name_or_path in torchaudio.pipelines.__all__:
            pipeline_type = "torchaudio"
            bundle = torchaudio.pipelines.__dict__[model_name_or_path]
            align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
            labels = bundle.get_labels()
            align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
        else:
            try:
                processor = Wav2Vec2Processor.from_pretrained(
                    model_name_or_path, cache_dir=model_dir, trust_remote_code=True
                )
                align_model = Wav2Vec2ForCTC.from_pretrained(
                    model_name_or_path, cache_dir=model_dir, trust_remote_code=True
                )
            except Exception as e:
                print(e)
                raise ValueError(
                    f'The chosen align_model "{model_name_or_path}" could not be found in '
                    f"huggingface or torchaudio"
                )
            pipeline_type = "huggingface"
            align_model = align_model.to(device)
            labels = processor.tokenizer.get_vocab()
            align_dictionary = {
                char.lower(): code
                for char, code in processor.tokenizer.get_vocab().items()
            }

        align_metadata = {
            "language": language_code,
            "dictionary": align_dictionary,
            "type": pipeline_type,
        }
        return align_model, align_metadata

    # ------------------------------------------------------------------ #
    #  单条音频对齐（与原版相同，仅接口不变）                                #
    # ------------------------------------------------------------------ #
    def align(
        self,
        transcript: Iterable[SingleSegment],
        model: torch.nn.Module,
        align_model_metadata: dict,
        audio: Union[str, np.ndarray, torch.Tensor],
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
        combined_progress: bool = False,
    ) -> AlignedTranscriptionResult:
        """Align phoneme recognition predictions to known transcription."""

        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        MAX_DURATION = audio.shape[1] / SAMPLE_RATE

        model_dictionary = align_model_metadata["dictionary"]
        model_lang = align_model_metadata["language"]
        model_type = align_model_metadata["type"]

        total_segments = len(transcript)
        segment_data: dict[int, SegmentData] = {}

        for sdx, segment in enumerate(transcript):
            if print_progress:
                base_progress = ((sdx + 1) / total_segments) * 100
                percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")

            num_leading = len(segment["text"]) - len(segment["text"].lstrip())
            num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
            text = segment["text"]

            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                per_word = text.split(" ")
            else:
                per_word = text

            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(text):
                char_ = char.lower()
                if model_lang not in LANGUAGES_WITHOUT_SPACES:
                    char_ = char_.replace(" ", "|")
                if cdx < num_leading:
                    pass
                elif cdx > len(text) - num_trailing - 1:
                    pass
                elif char_ in model_dictionary.keys():
                    clean_char.append(char_)
                    clean_cdx.append(cdx)
                else:
                    clean_char.append("*")
                    clean_cdx.append(cdx)

            clean_wdx = []
            for wdx, wrd in enumerate(per_word):
                if any([c in model_dictionary.keys() for c in wrd.lower()]):
                    clean_wdx.append(wdx)
                else:
                    clean_wdx.append(wdx)

            punkt_lang = PUNKT_LANGUAGES.get(model_lang, "english")
            try:
                sentence_splitter = nltk_load(f"tokenizers/punkt_tab/{punkt_lang}.pickle")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                sentence_splitter = nltk_load(f"tokenizers/punkt_tab/{punkt_lang}.pickle")
            sentence_spans = list(sentence_splitter.span_tokenize(text))

            segment_data[sdx] = {
                "clean_char": clean_char,
                "clean_cdx": clean_cdx,
                "clean_wdx": clean_wdx,
                "sentence_spans": sentence_spans,
            }

        aligned_segments: List[SingleAlignedSegment] = []

        for sdx, segment in enumerate(transcript):
            t1 = segment["start"]
            t2 = segment["end"]
            text = segment["text"]

            aligned_seg: SingleAlignedSegment = {
                "start": t1,
                "end": t2,
                "text": text,
                "words": [],
                "chars": None,
            }

            if return_char_alignments:
                aligned_seg["chars"] = []

            if len(segment_data[sdx]["clean_char"]) == 0:
                logger.warning(
                    f'Failed to align segment ("{segment["text"]}"): no characters found'
                )
                aligned_segments.append(aligned_seg)
                continue

            if t1 >= MAX_DURATION:
                logger.warning(
                    f'Failed to align segment ("{segment["text"]}"): start time > audio duration'
                )
                aligned_segments.append(aligned_seg)
                continue

            text_clean = "".join(segment_data[sdx]["clean_char"])
            tokens = [model_dictionary.get(c, -1) for c in text_clean]

            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(t2 * SAMPLE_RATE)

            waveform_segment = audio[:, f1:f2]
            if waveform_segment.shape[-1] < 400:
                lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
                waveform_segment = torch.nn.functional.pad(
                    waveform_segment, (0, 400 - waveform_segment.shape[-1])
                )
            else:
                lengths = None

            with torch.inference_mode():
                if model_type == "torchaudio":
                    emissions, _ = model(waveform_segment.to(device), lengths=lengths)
                elif model_type == "huggingface":
                    emissions = model(waveform_segment.to(device)).logits
                else:
                    raise NotImplementedError(f"Model type {model_type} not supported.")
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            blank_id = 0
            for char, code in model_dictionary.items():
                if char == "[pad]" or char == "<pad>":
                    blank_id = code

            trellis = get_trellis(emission, tokens, blank_id)
            path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)

            if path is None:
                logger.warning(
                    f'Failed to align segment ("{segment["text"]}"): backtrack failed'
                )
                aligned_seg["words"] = [
                    {"word": text, "start": aligned_seg["start"], "end": aligned_seg["end"]}
                ]
                aligned_segments.append(aligned_seg)
                continue

            char_segments = merge_repeats(path, text_clean)
            duration = t2 - t1
            num_frames = trellis.size(0)

            if num_frames <= 1:
                # 音频太短，模型只给了一个 emission frame，无法正常 forced align
                # 直接把整个 utterance 分给这个 token
                logger.warning(
                    f'Failed to align segment ("{segment["text"]}"): backtrack failed, durantion <= 1'
                )
                aligned_seg["words"] = [
                    {"word": text, "start": aligned_seg["start"], "end": aligned_seg["end"]}
                ]
                aligned_segments.append(aligned_seg)
                continue
            
            # ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1 + 1e-8)
            ratio = duration / (trellis.size(0) - 1)

            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in segment_data[sdx]["clean_cdx"]:
                    char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)

                char_segments_arr.append(
                    {
                        "char": char,
                        "start": start,
                        "end": end,
                        "score": score,
                        "word-idx": word_idx,
                    }
                )

                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    word_idx += 1
                elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                    word_idx += 1

            char_segments_arr = pd.DataFrame(char_segments_arr)

            aligned_subsegments = []
            char_segments_arr["sentence-idx"] = None
            for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
                curr_chars = char_segments_arr.loc[
                    (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)
                ]
                char_segments_arr.loc[
                    (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send),
                    "sentence-idx",
                ] = sdx2

                sentence_text = text[sstart:send]
                sentence_start = curr_chars["start"].min()
                end_chars = curr_chars[curr_chars["char"] != " "]
                sentence_end = end_chars["end"].max()
                sentence_words = []

                for word_idx in curr_chars["word-idx"].unique():
                    word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                    word_text = "".join(word_chars["char"].tolist()).strip()
                    if len(word_text) == 0:
                        continue

                    word_chars = word_chars[word_chars["char"] != " "]
                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

                    word_segment = {"word": word_text}
                    if not np.isnan(word_start):
                        word_segment["start"] = word_start.item()
                    if not np.isnan(word_end):
                        word_segment["end"] = word_end.item()
                    if not np.isnan(word_score):
                        word_segment["score"] = word_score.item()

                    sentence_words.append(word_segment)

                aligned_subsegments.append(
                    {
                        "text": sentence_text,
                        "start": sentence_start,
                        "end": sentence_end,
                        "words": sentence_words,
                    }
                )

                if return_char_alignments:
                    curr_chars = curr_chars[["char", "start", "end", "score"]]
                    curr_chars.fillna(-1, inplace=True)
                    curr_chars = curr_chars.to_dict("records")
                    curr_chars = [
                        {key: val for key, val in char.items() if val != -1}
                        for char in curr_chars
                    ]
                    aligned_subsegments[-1]["chars"] = curr_chars

            aligned_subsegments = pd.DataFrame(aligned_subsegments)
            aligned_subsegments["start"] = interpolate_nans(
                aligned_subsegments["start"], method=interpolate_method
            )
            aligned_subsegments["end"] = interpolate_nans(
                aligned_subsegments["end"], method=interpolate_method
            )
            agg_dict = {"text": " ".join, "words": "sum"}
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"
            aligned_subsegments = aligned_subsegments.groupby(
                ["start", "end"], as_index=False
            ).agg(agg_dict)
            aligned_subsegments = aligned_subsegments.to_dict("records")
            aligned_segments += aligned_subsegments

        word_segments: List[SingleWordSegment] = []
        for segment in aligned_segments:
            word_segments += segment["words"]

        return {"segments": aligned_segments, "word_segments": word_segments}

    def generate(self):
        total_batches = len(self.transcript_loader)

        for batch_idx, batch in enumerate(self.transcript_loader):
            waveforms   = batch["waveforms"]    # List[Tensor]  长度 = batch_size
            segments    = batch["segments"]     # List[List[SingleSegment]]
            audio_paths = batch["audio_paths"]  # List[str]

            if self.is_main:
                logger.info(
                    f"[rank {self.local_rank}] batch {batch_idx + 1}/{total_batches}"
                    f" — {len(audio_paths)} audio(s)"
                )

            for waveform, segment, audio_path in zip(waveforms, segments, audio_paths):
                if segment is None or len(segment) == 0:
                    logger.warning(
                        f'No valid segments found for audio "{audio_path}". Skipping alignment.'
                    )
                    continue
                result = self.align(
                    transcript=segment,
                    model=self.model,
                    align_model_metadata=self.metadata,
                    audio=waveform,
                    device=self.device,
                )

                item_segs = [
                    {
                        "start": s["start"],
                        "end": s["end"],
                        "text": s["text"],
                        "words": [
                            {
                                "start": w["start"],
                                "end": w["end"],
                                "word": w["word"],
                            }
                            for w in s["words"]
                        ],
                    }
                    for s in result["segments"]
                ]

                stem = (
                    pathlib.Path(audio_path).stem
                )
                output_path = self.output_dir / f"{stem}.jsonl"
                self._save_to_jsonl(
                    segments=item_segs,
                    audio_path=audio_path,
                    output_path=str(output_path),
                )

        # 所有卡都处理完自己的分片后做一次 barrier，保证 main 进程日志准确
        dist.barrier()
        if self.is_main:
            logger.info("All ranks finished.")

    def _save_to_jsonl(self, segments: list, audio_path: str, output_path: str):
        output_path = pathlib.Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            for segment in segments:
                words = segment["words"]
                if words and words[-1]["word"] in ERROR_TOKEN:
                    words = words[:-1]
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_path},
                            {"type": "text", "text": "transcrip it"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text_stream",
                                "text_stream": [
                                    [word["start"], word["end"], word["word"]]
                                    for word in words
                                ],
                            }
                        ],
                    },
                ]
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        logger.info(
            f"[rank {self.local_rank}] saved {len(segments)} segments → {output_path}"
        )


# ======================================================================= #
#  CTC 算法（与原版完全相同）                                               #
# ======================================================================= #

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis


def get_wildcard_emission(frame_emission, tokens, blank_id):
    assert 0 <= blank_id < len(frame_emission)
    tokens = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
    wildcard_mask = tokens == -1
    regular_scores = frame_emission[tokens.clamp(min=0).long()]
    max_valid_score = frame_emission.clone()
    max_valid_score[blank_id] = float("-inf")
    max_valid_score = max_valid_score.max()
    return torch.where(wildcard_mask, max_valid_score, regular_scores)


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        assert t > 0
        p_stay = emission[t - 1, blank_id]
        p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change
        t -= 1
        if changed > stayed:
            j -= 1
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1
    return path[::-1]


@dataclass
class Path:
    points: List[Point]
    score: float


@dataclass
class BeamState:
    token_index: int
    time_index: int
    score: float
    path: List[Point]


def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    T, J = trellis.size(0) - 1, trellis.size(1) - 1
    init_state = BeamState(
        token_index=J,
        time_index=T,
        score=trellis[T, J],
        path=[Point(J, T, emission[T, blank_id].exp().item())],
    )
    beams = [init_state]

    while beams and beams[0].token_index > 0:
        next_beams = []
        for beam in beams:
            t, j = beam.time_index, beam.token_index
            if t <= 0:
                continue
            p_stay = emission[t - 1, blank_id]
            p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]
            stay_score = trellis[t - 1, j]
            change_score = trellis[t - 1, j - 1] if j > 0 else float("-inf")

            if not math.isinf(stay_score):
                new_path = beam.path.copy()
                new_path.append(Point(j, t - 1, p_stay.exp().item()))
                next_beams.append(
                    BeamState(token_index=j, time_index=t - 1, score=stay_score, path=new_path)
                )
            if j > 0 and not math.isinf(change_score):
                new_path = beam.path.copy()
                new_path.append(Point(j - 1, t - 1, p_change.exp().item()))
                next_beams.append(
                    BeamState(token_index=j - 1, time_index=t - 1, score=change_score, path=new_path)
                )

        beams = sorted(next_beams, key=lambda x: x.score, reverse=True)[:beam_width]
        if not beams:
            break

    if not beams:
        return None

    best_beam = beams[0]
    t = best_beam.time_index
    j = best_beam.token_index
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        best_beam.path.append(Point(j, t - 1, prob))
        t -= 1
    return best_beam.path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2 - 1].time_index + 1, score)
        )
        i1 = i2
    return segments


def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


# ======================================================================= #
#  CLI                                                                      #
# ======================================================================= #

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--annotation-dir", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--output-dir", type=str, default="./results/alignments")
    parser.add_argument("--device", type=str, default="cuda")


def run():
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    alignment = Alignment(
        annotation_dir=args.annotation_dir,
        target_sr=args.target_sr,
        batch_size=args.batch_size,
        language_code=args.language,
        device=device,                      # 每卡绑定自己的 GPU
        model_name_or_path=args.model_name_or_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        local_rank=local_rank,
        world_size=world_size,
    )

    alignment.generate()

    dist.destroy_process_group()


if __name__ == "__main__":
    run()