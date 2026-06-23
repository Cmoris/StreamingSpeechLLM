import torch
import torchaudio
from types import SimpleNamespace
from modules import build_audio_tower
# from your_module import build_audio_tower

AUDIO_FILE = "/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audio_clips/A_gd/260_515.wav"


def load_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)

    # 转单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # (1, T) -> (T,)
    waveform = waveform.squeeze(0)

    return waveform


def test_audio_tower(model_name):
    print(f"\n===== Testing: {model_name} =====")

    cfg = SimpleNamespace(audio_tower=model_name)

    tower, _ = build_audio_tower(cfg, delay_load=False)
    tower.to("cuda")
    tower.eval()

    print("Hidden size:", tower.hidden_size)
    print("Device:", tower.device)
    print("Dtype:", tower.dtype)

    # ===== 读取真实音频 =====
    audio = load_audio(AUDIO_FILE)
    audio = audio.unsqueeze(0)  # (B=1, T)

    print("Input waveform shape:", audio.shape)

    # ===== 不同 encoder 适配 =====
    if "whisper" in model_name:
        processor = tower.audio_processor

        inputs = processor(
            [a.numpy() for a in audio],
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features  # (1, 80, T)
        breakpoint()
        print("Whisper input:", input_features.shape)

        output = tower(input_features)

    elif "hubert" in model_name:
        processor = tower.audio_processor

        inputs = processor(
            audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values

        print("HuBERT input:", input_values.shape)

        output = tower(input_values)

    else:  # zipformer
        processor = tower.audio_processor
        inputs = processor(
            audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values
        padding_mask = (input_values != processor.padding_value)
        print("Zipformer input:", input_values.shape)
        output = tower(input_values, padding_mask)

    # ===== 输出 =====
    if isinstance(output, list):
        for i, o in enumerate(output):
            print(f"[{i}] shape:", o.shape)
    else:
        print("Output shape:", output.shape)

    print("Output dtype:", output.dtype)
    print("Done ✔")


if __name__ == "__main__":
    TEST_MODELS = [
        "openai/whisper-large-v3",
        "reazon-research/japanese-hubert-base-k2",
        "reazon-research/japanese-zipformer-base-k2",  # 如果加载失败可以换
    ]

    for m in TEST_MODELS:
        try:
            test_audio_tower(m)
        except Exception as e:
            print(f"❌ Failed on {m}: {e}")