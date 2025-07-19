import torch
import argparse
from model.transformer import StemTransformerClassifier
from torch.utils.data import DataLoader
from utils.collate import collate
from utils.plot import plot_chunk_level_accuracy_per_class, plot_stem_level_accuracy_per_class, plot_confusion_matrix
from tqdm import tqdm
from collections import defaultdict, Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and plot classification results.")
    parser.add_argument('--model', type=str, default="model.pth", help="Path to model weights")
    parser.add_argument('--data', type=str, default="./export/data.pth", help="Path to dataset .pth file")
    parser.add_argument('--plot', action='store_true', help="If set, generate and save plots")
    args = parser.parse_args()

    # Load model
    model = StemTransformerClassifier()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Load dataset
    dataset = torch.load(args.data, weights_only=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

    all_preds = []
    all_labels = []
    all_stem_names = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, leave=False)
        for x, pad_mask, y, stem_name in progress_bar:
            x = x.to(device)
            pad_mask = pad_mask.to(device)
            output = model(x, pad_mask)
            predictions = torch.argmax(output, dim=1)

            all_preds.append(predictions.cpu())
            all_labels.append(y.cpu())
            all_stem_names.append(stem_name)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_stem_names = [s for batch in all_stem_names for s in batch]

    chunk_accuracy = (all_preds == all_labels).float().mean().item()
    print(f"\nChunk-level Accuracy: {chunk_accuracy * 100:.2f}%")

    stem_preds = defaultdict(list)
    stem_labels = {}

    for pred, label, stem in zip(all_preds, all_labels, all_stem_names):
        stem_preds[stem].append(pred.item())
        stem_labels[stem] = label.item()

    correct = 0
    total = 0

    for stem, preds in stem_preds.items():
        majority_class = Counter(preds).most_common(1)[0][0]
        true_class = stem_labels[stem]
        if majority_class == true_class:
            correct += 1
        total += 1

    stem_accuracy = correct / total
    print(f"Stem-level Accuracy: {stem_accuracy * 100:.2f}%")

    if args.plot:
        plot_stem_level_accuracy_per_class(all_preds, all_labels, all_stem_names)
        plot_chunk_level_accuracy_per_class(all_preds, all_labels)
        plot_confusion_matrix(all_preds, all_labels, class_names=None, normalize=False)

