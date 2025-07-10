# etts/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ETTSData(Dataset):
    def __init__(self, folder):
        self.files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        phonemes = torch.tensor(data["phonemes"], dtype=torch.long)
        embedding = torch.tensor(data["embedding"], dtype=torch.float)
        mel = torch.tensor(data["mel"], dtype=torch.float)
        return phonemes, embedding, mel
