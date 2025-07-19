import librosa
import numpy as np
import torch
import torch.nn as nn

class Features(nn.Module):
    def __init__(self, sample_rate=44100, n_mfcc=8, n_fft=2048, hop_length=512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def forward(self, waveforms):
        # waveform: Tensor of shape (1, N)
        features = []
        for i in range(len(waveforms)):
            y = waveforms[i].squeeze(0).numpy()

            # Compute MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )  # shape: (n_mfcc, T)

            # Compute f0 (using librosa.pyin for robustness)
            f0 = librosa.yin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
            )  # f0 shape: (T,)

            # Replace unvoiced frames with 0 or NaN
            f0 = np.nan_to_num(f0)  # (T,)

            # Normalize (optional)
            f0 = (f0 - f0.mean()) / (f0.std() + 1e-8)

            # Reshape and concatenate
            f0 = f0[np.newaxis, :]          # (1, T)
            stacked_features = np.vstack([mfcc, f0])  # (n_mfcc + 1, T)

            features.append(torch.from_numpy(stacked_features).float())

        return features 

