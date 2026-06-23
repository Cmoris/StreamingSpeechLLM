from torch.utils.data import Dataset, DataLoader
from transformers import logging
from sentence_transformers import SentenceTransformer

import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from typing import List, Dict, TypedDict

class SingleSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech.
    """

    start: float
    end: float
    text: str

logger = logging.get_logger(__name__)


class TranscriptDataset(Dataset):
    def __init__(self, annotation_dir: str):
        self.folder_path = Path(annotation_dir)
        self.items = []  # each item: (audio_path, segments)

        self._load_jsonl_files()

    def _load_jsonl_files(self):
        jsonl_files = sorted(self.folder_path.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {self.folder_path}")

        for jsonl_file in jsonl_files:
            audio_path = None
            segments :List[SingleSegment] = []

            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[:-1]: # skip the last line which stores the seek indices
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if audio_path is None:
                        audio_path = record["audio_path"]

                    segments.append(SingleSegment(
                        text=record["text"],
                        start=record["start"],
                        end=record["end"]
                    ))

            if audio_path is not None and segments:
                self.items.append((audio_path, segments))

    def __getitem__(self, idx):
        audio_path, segments = self.items[idx]

        return audio_path, segments
    
    def __len__(self):
        return len(self.items)
    
    
class TranscriptDatasetForTXT(Dataset):
    def __init__(self, annotation_dir: str):
        super().__init__()
        self.folder_path = Path(annotation_dir)
        self.items = []  # each item: (audio_path, segments)
        
        self._load_txt_files()
        
    def _load_txt_files(self):
        txt_files = sorted(self.folder_path.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.folder_path}")

        for txt_file in txt_files:
            audio_path = txt_file.parent / (txt_file.stem + ".wav")  # assuming audio files have the same name as txt files but with .wav extension
            segments :List[SingleSegment] = []

            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # parse the line to extract audio_path, text, start, end
                    # this depends on the format of the txt file, which is not specified here
                    # assuming a format like: audio_path \t start \t end \t text
                    parts = line.split("\t")
                    if len(parts) != 3:
                        continue
                    start, end, text = parts
                    segments.append(SingleSegment(
                        text=text,
                        start=float(start),
                        end=float(end)
                    ))

            if audio_path is not None and segments:
                self.items.append((audio_path, segments))
                
    def __getitem__(self, idx):
        audio_path, segments = self.items[idx]

        return str(audio_path), segments
    
    def __len__(self):
        return len(self.items)


def get_embedding(text)
    # Load the model
    # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # together with setting `padding_side` to "left":
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    # The queries and documents to embed
    quries = [text]

    # Encode the queries and documents. Note that queries benefit from using a prompt
    # Here we use the prompt called "query" stored under `model.prompts`, but you can
    # also pass your own prompt via the `prompt` argument
    query_embeddings = model.encode(queries)


    
def is_back_channel(segment: SingleSegment, threshold: float=0.5) -> bool:
    """
    Heuristic to determine if a segment is a back-channel response.
    This can be based on the text content, duration, or other features.
    For simplicity, we check if the text contains common back-channel words and if the duration is short.
    """
    # back_channel_keywords = ["え", "ねえ", "そ", "あ", "うん", "うーん", "ああ", "はい", "へえ", "ほう", "ふーん"]  # common Japanese back-channel words
    text_lower = segment["text"].lower()
    duration = segment["end"] - segment["start"]

    if duration < threshold and len(text_lower) < 5:  # Example threshold for text length
        
        return True
    return False

if __name__ == "__main__":
    A_gd = "/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/A_gd"
    B_gd = "/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/B_gd"
    A_all = "/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/A_all"
    B_all = "/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/B_all"
    dataset = TranscriptDatasetForTXT(annotation_dir=A_gd)
    for audio_path, segments in tqdm(dataset):
        for segment in segments:
            if is_back_channel(segment, threshold=2):
                print(f"Back-channel detected in {audio_path}: [{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")