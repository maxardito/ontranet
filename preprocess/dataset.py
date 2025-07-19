import os
import librosa
import torch
from torch.utils.data import Dataset

class StemAudioDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.filepaths = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]
        # Extract labels from filenames like 'xxx-labelname.mp3'
        self.label_map = {'bass': 0, 'drums': 1, 'other': 2, 'vocal': 3}
        self.transform = transform

        # Precompute onset chunks for all files and store (chunk, label)
        self.data = []  # list of (waveform_chunk, label)
        for filename in self.filepaths:
            filepath = os.path.join(self.folder_path, filename)
            y, sr = librosa.load(filepath, sr=None)
            waveform = torch.tensor(y).unsqueeze(0)
            label_str = filename.split('-')[1][:-4]
            label = self.label_map[label_str]
            print("File name: ", filename, " -- Label string: ", label_str, " -- Label: ", label)

            if self.transform:
                chunks = self.transform(waveform)  # Expected to return list of chunks
                # Append each chunk + label as one item
                for chunk in chunks:
                    self.data.append((chunk, label, filename))
            else:
                # If no transform, just store full waveform as one chunk
                self.data.append((waveform, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
