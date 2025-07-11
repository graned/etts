import torch
import os
import sys
import matplotlib.pyplot as plt
import torchvision.utils as vutils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from models.etts_model import ETTSModel
from data.etts_dataloader import ETTSDataloader
from utils.phoneme_dictionary import PhonemeDictionary
from utils.embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor
from utils.manifest_builder import ManifestBuilder

# 📦 Load components
PHONEME_DICT_PATH = "train/dictionaries/phoneme_dict.json"
MANIFEST_PATH = "train/manifests/etts_manifest.json"
CHECKPOINT_DIR = "train/checkpoints"
SAMPLES_PATH = "train/samples"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train():
    writer = SummaryWriter(log_dir="runs/etts_experiment")  # TensorBoard writer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Manifest
    samples_path = "train/samples"
    manifest_builder = ManifestBuilder(samples_path)
    manifest_builder.build()
    manifest_path = "train/manifests/etts_manifest.json"
    manifest_builder.save(manifest_path)

    # 🧠 Load phoneme dictionary
    phoneme_dict = PhonemeDictionary(lang="en-us", vocab_path=PHONEME_DICT_PATH)

    # Initialize utilities
    embedding_extractor = EmbeddingExtractor()
    mel_extractor = MelExtractor()

    # 🧬 Load data + preprocessing tools
    dataloader_builder = ETTSDataloader(
        phoneme_dict=phoneme_dict,
        embedding_extractor=embedding_extractor,
        mel_extractor=mel_extractor,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Use 0 for simpler debugging
    )
    dataloader = dataloader_builder.load(manifest_path)

    # Get one batch to estimate upsample factor
    phonemes, phoneme_lengths, speaker_embs, mels, mel_lengths = next(iter(dataloader))

    avg_phoneme_len = phonemes.shape[1]  # number of phoneme steps
    avg_mel_len = mels.shape[2]  # number of mel time steps

    upsample_factor = round(avg_mel_len / avg_phoneme_len)
    print(f"Estimated upsample factor: {upsample_factor}")

    # 🏗️ Model, loss, optimizer
    model = ETTSModel(
        num_phonemes=phoneme_dict.get_num_phonemes(), upsample_factor=upsample_factor
    )
    # Loss function and optimizer
    criterion = nn.MSELoss()  # for mel spectrogram regression
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 50
    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"\n🌱 Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        outputs = None
        for batch_idx, (
            phonemes,
            phoneme_lengths,
            speaker_embs,
            mels,
            mel_lengths,
        ) in enumerate(dataloader):
            # phonemes = phonemes.to(device)
            speaker_emb = speaker_embs.to(device)
            # mels = mels.to(device)

            optimizer.zero_grad()
            outputs = model(phonemes, speaker_emb)
            loss = criterion(outputs, mels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(
                        f"GradientsHist/{name}", param.grad.norm(), batch_idx
                    )

            optimizer.step()

            for group in optimizer.param_groups:
                writer.add_scalar("Learning Rate", group["lr"], batch_idx)

            total_loss += loss.item()

            writer.add_scalar("Loss/train", loss.item(), batch_idx)
            # Log every few steps
            if batch_idx % 10 == 0:
                visualize_mel(writer, mels[0], batch_idx, "Mel/Target")
                visualize_mel(writer, outputs[0], batch_idx, "Mel/Generated")

        avg_loss = total_loss / len(dataloader)
        print(f"📉 Average Loss: {avg_loss:.6f}")

        # 💾 Save checkpoint
        model_path = os.path.join(CHECKPOINT_DIR, f"etts_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved checkpoint: {model_path}")

        # 🧪 Optional: save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )
    writer.close()


def visualize_mel(writer, mel_tensor, step, tag="mel_spectrogram"):
    mel = mel_tensor.clone().detach().cpu()
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-5)  # Normalize to [0, 1]
    mel = mel.unsqueeze(0)  # Add batch dimension
    writer.add_image(tag, mel, global_step=step)


if __name__ == "__main__":
    train()
