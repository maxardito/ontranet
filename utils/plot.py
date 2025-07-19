import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Use a consistent, readable theme
sns.set_theme(style="whitegrid", font_scale=1.2)

CLASS_NAMES = ['bass', 'drums', 'other', 'vocal']

def plot_stem_level_accuracy_per_class(all_preds, all_labels, all_stem_names):
    from collections import defaultdict, Counter

    stem_preds = defaultdict(list)
    stem_labels = {}

    for pred, label, stem in zip(all_preds, all_labels, all_stem_names):
        stem_preds[stem].append(pred.item())
        stem_labels[stem] = label.item()

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for stem, preds in stem_preds.items():
        true_class = stem_labels[stem]
        majority_class = Counter(preds).most_common(1)[0][0]
        total_per_class[true_class] += 1
        if majority_class == true_class:
            correct_per_class[true_class] += 1

    classes = sorted(total_per_class.keys())
    accs = [correct_per_class[c] / total_per_class[c] for c in classes]
    class_labels = [CLASS_NAMES[c] for c in classes]

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(class_labels))
    barplot = sns.barplot(x=class_labels, y=accs, palette=colors)

    for i, acc in enumerate(accs):
        barplot.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

    plt.title("Stem-Level Accuracy per Class")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./plots/stem_accuracy.png")
    plt.close()

def plot_chunk_level_accuracy_per_class(all_preds, all_labels):
    from collections import defaultdict

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        label = label.item()
        pred = pred.item()
        total_per_class[label] += 1
        if pred == label:
            correct_per_class[label] += 1

    classes = sorted(total_per_class.keys())
    accs = [correct_per_class[c] / total_per_class[c] for c in classes]
    class_labels = [CLASS_NAMES[c] for c in classes]

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(class_labels))
    barplot = sns.barplot(x=class_labels, y=accs, palette=colors)

    for i, acc in enumerate(accs):
        barplot.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontsize=10)

    plt.title("Chunk-Level Accuracy per Class")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./plots/chunk_accuracy.png")
    plt.close()

def plot_confusion_matrix(all_preds, all_labels, class_names=None, normalize=False):
    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()

    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True,
        cbar=True
    )

    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("./plots/confusion_matrix.png")
    plt.close()

