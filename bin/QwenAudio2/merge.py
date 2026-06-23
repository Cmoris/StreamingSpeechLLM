import json
import argparse
from pathlib import Path
from typing import List
from collections import defaultdict

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/yizhang/Moris/StreamingSpeechLLM/bin/QwenAudio2/results_asr/qwen2audio_asrl2_chunk2s_lora16",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/yizhang/Moris/StreamingSpeechLLM/bin/QwenAudio2/results_asr"
    )


def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()
    
def read_data(path:Path):
    with open(path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        yield line           

def merge_rank_jsonls(
    rank_files: List[Path],
    output_path: Path,
):
    segments = []
    for fp in rank_files:
        for seg in read_data(fp):
            segments.append(seg)

    if len(segments) == 0:
        raise RuntimeError("No segments to merge.")
    
    segments = set(segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = []
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            seg = json.loads(seg)
            line = json.dumps(seg, ensure_ascii=False) + "\n"
            lengths.append(len(line))
            f.write(line)

    print(f"[OK] merged {len(rank_files)} files -> {output_path}")
    
def merge_all_by_stem(data_dir: str, output_dir: str):
    output_dir = Path(output_dir)
    Path.mkdir(output_dir, exist_ok=True)
    
    data_dir = Path(data_dir)
    name = data_dir.name
    output_path = output_dir / f"{name}.jsonl"
    rank_files = [x for x in data_dir.glob("*_rank*.jsonl")]
    merge_rank_jsonls(
        rank_files=rank_files,
        output_path=output_path,
    )
        

def run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    merge_all_by_stem(args.data_dir, args.output_dir)

if __name__ == "__main__":
    run()