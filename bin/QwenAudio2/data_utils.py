import numpy as np
import librosa
import soundfile as sf

import torch

from constants import DEFAULT_SAMPLE_RATE


def read_audio(ele: dict):
    path = ele["audio"]

    # 1. 读取音频，不走 torchcodec
    wav, orig_sr = sf.read(path, dtype="float32")

    # stereo / multi-channel -> mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    audio_duration = len(wav) / orig_sr

    audio_start = ele.get("audio_start", None)
    audio_end = ele.get("audio_end", None)

    if audio_start is None:
        audio_start = 0.0
    if audio_end is None:
        audio_end = audio_duration

    # 防止越界
    audio_start = max(0.0, float(audio_start))
    audio_end = min(float(audio_end), audio_duration)

    if audio_end <= audio_start:
        # 避免空音频
        audio_end = min(audio_start + 1.0 / orig_sr, audio_duration)

    # 2. 先在原采样率下裁剪
    start_sample = int(round(audio_start * orig_sr))
    end_sample = int(round(audio_end * orig_sr))
    clip = wav[start_sample:end_sample]

    # 3. 重采样到 DEFAULT_SAMPLE_RATE
    if orig_sr != DEFAULT_SAMPLE_RATE:
        clip = librosa.resample(
            clip,
            orig_sr=orig_sr,
            target_sr=DEFAULT_SAMPLE_RATE,
        )
        audio_sr = DEFAULT_SAMPLE_RATE
    else:
        audio_sr = orig_sr

    # 4. 构造 clip_pts，对应重采样后的每个 sample 的原始时间戳
    nframes = len(clip)
    clip_pts = audio_start + np.arange(nframes) / DEFAULT_SAMPLE_RATE

    clip = torch.from_numpy(clip).float()

    return clip, clip_pts, audio_sr


def safe_chunk(wav: torch.Tensor, start: int, end: int, chunk_samples: int) -> torch.Tensor:
    chunk = wav[start:end]
    if len(chunk) < chunk_samples:
        pad = make_dummy_audio(chunk_samples - len(chunk))  
        chunk = torch.cat([chunk, pad])
    return chunk

def make_dummy_audio(num_samples: int, noise_scale: float = 1e-4) -> torch.Tensor:
    """用极小噪声代替静音，避免被 feature extractor 当成 padding 截断"""
    return torch.randn(num_samples) * noise_scale

def pad_audio_to_min_len(wav: torch.Tensor, min_samples: int) -> torch.Tensor:
    if wav.numel() >= min_samples:
        return wav
    pad = make_dummy_audio(min_samples - wav.numel()).to(wav.device)
    return torch.cat([wav, pad], dim=0)