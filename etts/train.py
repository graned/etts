# etts/train.py
import torch
from torch.utils.data import DataLoader
from etts.model import ETTSTransformer
from etts.dataset import ETTSData
import torch.nn.functional as F


def train():
    dataset = ETTSData("data/train")  # Folder with .npz files
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ETTSTransformer().to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        for phonemes, embedding, mel in loader:
            pred = model(phonemes, embedding)
            loss = F.mse_loss(pred, mel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
