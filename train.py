import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.collate import collate
from model.transformer import StemTransformerClassifier
from tqdm import tqdm

def train_model(model, dataloader, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for x, pad_mask, y in progress_bar:
            x = x.to(device)
            pad_mask = pad_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, pad_mask)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()

            avg_loss = total_loss / total_samples
            accuracy = correct / total_samples
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Stem Transformer model.")
    parser.add_argument("--data", "-d", type=str, default="data.pth",
                        help="Path to the input .pth dataset file")
    args = parser.parse_args()

    dataset = torch.load(args.data, weights_only=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StemTransformerClassifier().to(device)

    train_model(model, dataloader)

    torch.save(model.state_dict(), "model.pth")


