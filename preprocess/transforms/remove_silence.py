import torch
import numpy as np
import librosa
from torch import nn

class RemoveSilence(nn.Module):
    def __init__(self, threshold=0.01, frame_length=2048, hop_length=512):
        super().__init__()
        self.threshold = threshold
        self.frame_length = frame_length
        self.hop_length = hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (1, time) mono waveform tensor
        returns: waveform with silent parts removed
        """
        y = waveform.squeeze().cpu().numpy()

        rms = librosa.feature.rms(y=y,
                                  frame_length=self.frame_length,
                                  hop_length=self.hop_length)[0]

        non_silent_indices = [i for i, e in enumerate(rms) if e > self.threshold]

        if not non_silent_indices:
            return torch.zeros_like(waveform)

        non_silent_samples = []
        for i in non_silent_indices:
            start = i * self.hop_length
            end = start + self.frame_length
            non_silent_samples.append(y[start:end])

        y_nonsilent = np.concatenate(non_silent_samples)
        y_nonsilent = torch.tensor(y_nonsilent, dtype=waveform.dtype).unsqueeze(0)

        return y_nonsilent

