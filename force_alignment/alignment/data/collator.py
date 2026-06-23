import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict, Tuple, Any

class DataCollatorForTranscripts:

    pad_value: float = 0.0

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, str, List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:

        waveforms, audio_paths, segments = zip(*batch)
        # Record original lengths (number of samples, channel-agnostic)
        waveform_lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)

        padded = pad_sequence(
            waveforms,
            batch_first=True,
            padding_value=self.pad_value,
        )  # [B, T_max]

        return {
            "waveforms":        padded,
            "waveform_lengths": waveform_lengths,
            "audio_paths":      list(audio_paths),
            "segments":         list(segments),
        }