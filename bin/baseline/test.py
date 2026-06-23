import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import transformers
from dataset import DualChannelConvDataset

from model.submodels.modules import build_audio_tower

@dataclass
class DataArguments:
    annotation_dir: str   = field(default="/n/work6/yizhang/Moris/zoom2025/pretrain_labels/l3_conv_train_with_backchannel")
    audio_root_a:   str   = field(default="/n/work6/yizhang/Moris/zoom2025/audios/A_all")
    audio_root_b:   str   = field(default="/n/work6/yizhang/Moris/zoom2025/audios/B_all")
    context_length: int   = field(default=2)
    sample_rate:    int   = field(default=16000)
    query:          Optional[str] = field(default=None)

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    processor: transformers.AutoProcessor,
    data_args: DataArguments,
) -> Dict:
    annotation_paths = [
        str(x) for x in Path(data_args.annotation_dir).glob("*.jsonl")
    ]
    if not annotation_paths:
        raise FileNotFoundError(
            f"No .jsonl files found in annotation_dir={data_args.annotation_dir!r}"
        )

    dataset = DualChannelConvDataset(
        annotation_paths=annotation_paths,
        processor=processor,
        tokenizer=tokenizer,
        audio_root_a=data_args.audio_root_a,
        audio_root_b=data_args.audio_root_b,
        context_length=data_args.context_length,
        sample_rate=data_args.sample_rate,
        query=data_args.query,
    )

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=dataset.data_collator,
    )


def test_audio_tower_conv_simple(
    model_name,
    tokenizer,
    processor,
    data_args,
    batch_size=2,
    log_path="bad_batches.log",
):
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )

    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    cfg = SimpleNamespace(audio_tower=model_name)
    tower, _ = build_audio_tower(cfg, delay_load=False)
    tower = tower.to("cuda").eval()

    conv = nn.Conv1d(
        in_channels=tower.hidden_size,
        out_channels=tower.hidden_size,
        kernel_size=2,
        stride=2,
    ).to("cuda").eval()

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    good = 0
    bad = 0

    for batch_idx, batch in enumerate(dataloader):
        try:
            audios = batch["audios"]

            with torch.no_grad():
                out = tower(**audios)

                # 兼容不同输出格式
                if hasattr(out, "last_hidden_state"):
                    feat = out.last_hidden_state
                elif isinstance(out, (tuple, list)):
                    feat = out[0]
                else:
                    feat = out

                # feat: [B, T, C] -> [B, C, T]
                feat = feat.transpose(1, 2)

                # 测试能不能过 conv
                conv_out = conv(feat)

            good += 1

        except Exception as e:
            bad += 1

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"bad batch idx: {batch_idx}\n")
                f.write(f"error: {repr(e)}\n")
                f.write("texts:\n")

                f.write(batch["texts"])
                f.write("\n\n")


            print(f"bad batch {batch_idx}, saved texts to {log_path}")

    print("finished")
    print("good:", good)
    print("bad:", bad)
    
if __name__ == "__main__":
    data_args = DataArguments()
    from transformers import Qwen2AudioProcessor, AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained("reazon-research/japanese-zipformer-base-k2")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_audio_tower_conv_simple(
    model_name="reazon-research/japanese-zipformer-base-k2",
    tokenizer=tokenizer,
    processor=processor,
    data_args=data_args,
    batch_size=1,
    log_path="logs/bad_batches.log",
)