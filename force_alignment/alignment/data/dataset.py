from torch.utils.data import Dataset, DataLoader
from transformers import logging, AutoProcessor
from torchcodec.decoders import AudioDecoder

import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from typing import List, Dict

from .schema import SingleSegment

logger = logging.get_logger(__name__)

def read_audio(ele: dict, sampling_rate: int):
    audio_decoder = AudioDecoder(source=ele['Audio Path'], sample_rate=sampling_rate)
    audio_sr = audio_decoder.metadata.sample_rate
    audio_duration = audio_decoder.metadata.duration_seconds_from_header
    total_frames = int(audio_duration*audio_sr)
    audio_pts = np.linspace(1/audio_sr, audio_duration, total_frames)
    audio_start = ele.get("start", None)
    audio_end = ele.get("end", None)
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
        
    nframes = int(total_frames/audio_sr*sampling_rate)
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip_pts = audio_pts[clip_idxs]
    clip = audio_decoder.get_samples_played_in_range(start_seconds=audio_start, stop_seconds=audio_end+1/sampling_rate).data.squeeze(0)
    
    return clip, clip_pts, audio_sr

class TranscriptDataset(Dataset):
    """
    Dataset for loading audio files with their transcription segments from JSONL files.

    Args:
        folder_path (str): Path to the folder containing .jsonl files.
        sample_rate (int): Target sample rate for audio resampling. Default: 16000.

    Each item returns:
        - waveform (Tensor): Full audio waveform, shape [channels, samples]
        - audio_path (str): The audio file path string from the JSONL
        - segments (list): List of dicts with keys 'text', 'start', 'end'
    """

    def __init__(self, annotation_dir: str, target_sr: int = 16000):
        self.folder_path = Path(annotation_dir)
        self.target_sr = target_sr
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
                for line in lines: # skip the last line which stores the seek indices
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

        audio_decoder = AudioDecoder(audio_path, sample_rate=self.target_sr, num_channels=1)
        waveform = audio_decoder.get_all_samples().data.squeeze(0)

        return waveform, str(audio_path), segments
    
    def __len__(self):
        return len(self.items)
    
    
class TranscriptDatasetForTXT(Dataset):
    def __init__(self, annotation_dir: str, target_sr: int = 16000):
        super().__init__()
        self.folder_path = Path(annotation_dir)
        self.target_sr = target_sr
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

        audio_decoder = AudioDecoder(str(audio_path), sample_rate=self.target_sr, num_channels=1)
        waveform = audio_decoder.get_all_samples().data.squeeze(0)

        return waveform, str(audio_path), segments
    
    def __len__(self):
        return len(self.items)
    
def test_transcript_dataset():
    from collator import DataCollatorForTranscripts
    collator = DataCollatorForTranscripts()
    dataset = TranscriptDataset(annotation_dir="/n/work6/yizhang/Moris/zoom2025/asr_labels")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for batch in tqdm(dataloader):
        pass
        
def test_transcript_txt_dataset():
    from collator import DataCollatorForTranscripts
    collator = DataCollatorForTranscripts()
    dataset = TranscriptDatasetForTXT(annotation_dir="/mnt/nvme/workspaces/muyun/Dataset/zoom2025/audios/A_gd")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    for batch in tqdm(dataloader):
        pass
    

if __name__ == "__main__":
    test_transcript_dataset()