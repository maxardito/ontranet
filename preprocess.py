import argparse
import torch
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T
import soundfile as sf

from preprocess.dataset import StemAudioDataset
from preprocess.transforms.remove_silence import RemoveSilence
from preprocess.transforms.onsets import SplitOnsets
from preprocess.transforms.features import Features
from preprocess.compose import Compose

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from stem audio files.")
    parser.add_argument("--input", "-i", type=str, default="./dataset/stems",
                        help="Path to the input stem folder.")
    parser.add_argument("--output", "-o", type=str, default="./export/data.pth",
                        help="Path to save the extracted features.")

    args = parser.parse_args()

    stem_folder = args.input
    output_path = args.output

    remove_silence = RemoveSilence(threshold=0.01)
    split_onsets = SplitOnsets()
    extract_features = Features(sample_rate=44100, n_mfcc=8, hop_length=512)
    transform = Compose([remove_silence, split_onsets, extract_features])

    dataset = StemAudioDataset(stem_folder, transform=transform)

    torch.save(dataset.data, output_path)

