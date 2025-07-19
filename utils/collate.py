import torch
import torch.nn.functional as F

def collate(batch):
    """
    Pads sequences in the time dimension and returns:
    - padded [batch, max_len, 9]
    - padding_mask [batch, max_len]
    - labels [batch]
    """
    features, labels, stem_name = zip(*batch)  # each f: [9, n_i]
    lengths = [f.shape[1] for f in features]
    max_len = max(lengths)

    # Pad in time dimension, which is dim=1 → transpose to [n_i, 9]
    padded = torch.stack([
        F.pad(f.T, (0, 0, 0, max_len - f.shape[1]))  # f.T is [n_i, 9]
        for f in features
    ])  # [batch, max_len, 9]

    padding_mask = torch.tensor([
        [False] * l + [True] * (max_len - l)
        for l in lengths
    ])  # [batch, max_len]

    labels = torch.tensor(labels)
    return padded, padding_mask, labels, stem_name

