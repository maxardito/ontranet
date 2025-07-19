import torch
import numpy as np
import librosa
from torch import nn
from typing import List

class SplitOnsets(nn.Module):
    def __init__(self, sr=44100, hop_length=512, backtrack=False):
        """
        sr: sample rate of the waveform
        hop_length: hop length used for onset detection
        backtrack: whether to backtrack onsets to nearest preceding minimum energy
        """
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.backtrack = backtrack

    def forward(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        waveform: (1, time) mono waveform tensor
        returns: list of waveform chunks split on detected onsets
        """
        y = waveform.squeeze().cpu().numpy()

        # Detect onsets in frames
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            units='frames'
        )

        # Convert frame indices to sample indices
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=self.hop_length)
        # Add final sample index as end of audio
        onset_samples = np.append(onset_samples, len(y))

        chunks = []
        for i in range(len(onset_samples) - 1):
            start = onset_samples[i]
            end = onset_samples[i+1]
            chunk = y[start:end]
            chunk_tensor = torch.tensor(chunk, dtype=waveform.dtype).unsqueeze(0)
            chunks.append(chunk_tensor)

        return chunks

