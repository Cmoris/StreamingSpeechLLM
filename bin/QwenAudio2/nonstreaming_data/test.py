import json
from pathlib import Path
import sys
import tqdm
from torch.utils.data import DataLoader
from transformers import Qwen2AudioProcessor

from conv_model_dataset import DualChannelConvDataset
sys.path.append("../")
from constants import (TS_TOKEN, TE_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN,
                       SPEAKER_TOKENS, STREAMING_CONT, DEFAULT_CHUNK_SECS, DEFAULT_SAMPLE_RATE)

DEFAULT_CHUNK_SECS = 0.5

dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_test_with_backchannel"
output_dir = "/home/yizhang/Moris/StreamingSpeechLLM/bin/QwenAudio2/data"
output_path = Path(output_dir) / (Path(dir).name + f"_groundtruth_{DEFAULT_CHUNK_SECS}s.jsonl")

proc = Qwen2AudioProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    trust_remote_code=True
)

audio_token_id = proc.tokenizer("<|AUDIO|>").input_ids[0]

ds = DualChannelConvDataset(
        annotation_paths=[str(path) for path in Path(dir).glob("*.jsonl")],
        processor=proc,
        audio_root_a="/n/work6/yizhang/Moris/zoom2025/audios/A_gd",
        audio_root_b="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
    )

print(f"Dataset length: {len(ds)}")

loader = DataLoader(
    ds,
    batch_size=1,
    num_workers=8,
    shuffle=False,
    collate_fn=ds.data_collator
)

with open(output_path, "w", encoding="utf-8") as f:
    for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
        target_ids = batch["input_ids"].clone()
        target_ids[target_ids == -100] = proc.tokenizer.pad_token_id

        refs = proc.batch_decode(
            target_ids,
            skip_special_tokens=True
        )

        record = {
            "refs": refs,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")