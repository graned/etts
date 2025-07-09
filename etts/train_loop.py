# In your Colab notebook cell, first install dependencies:
!pip install torch torchvision torchaudio phonemizer librosa soundfile tqdm

# Mount Google Drive to save/load checkpoints
from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import json
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.notebook import tqdm
from tts_dataset import TTSDataSet
from collate_fn import collate_fn
from tts_model import TTSModel
from phoneme_vocab import get_vocab_size

def load_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


train_data = load_dataset("data/train_manifest.json")
val_data = load_dataset("data/val_manifest.json")

train_dataset = TTSDataSet(train_data)
val_dataset = TTSDataSet(val_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TTSModel(vocab_size=get_vocab_size()).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()

# Set checkpoint directory in your Google Drive
CHECKPOINT_DIR = "/content/drive/MyDrive/tts_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            phonemes = batch["phonemes"].to(device)
            speaker_emb = batch["speaker_embedding"].to(device)
            mel_target = batch["mel"].to(device)

            mel_pred, mel_postnet = model(phonemes, speaker_emb)
            loss = criterion(mel_pred, mel_target) + criterion(mel_postnet, mel_target)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

NUM_EPOCHS = 20
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch in loop:
        phonemes = batch["phonemes"].to(device)
        speaker_emb = batch["speaker_embedding"].to(device)
        mel_target = batch["mel"].to(device)

        optimizer.zero_grad()
        mel_pred, mel_postnet = model(phonemes, speaker_emb)

        loss = criterion(mel_pred, mel_target) + criterion(mel_postnet, mel_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    val_loss = validate(model, val_loader, criterion, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, os.path.join(CHECKPOINT_DIR, "best_model.pth"))

