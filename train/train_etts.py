import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch import nn, optim
from models.etts_model import ETTSModel
from data.etts_dataloader import ETTSDataloader
from utils.phoneme_dictionary import PhonemeDictionary
from utils.embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor
from utils.manifest_builder import ManifestBuilder


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build Manifest
    samples_path = "train/samples"
    manifest_builder = ManifestBuilder(samples_path)
    manifest_builder.build()
    manifest_path = "train/manifests/etts_manifest.json"
    manifest_builder.save(manifest_path)

    # Load phoneme dictionary
    dict_path = "train/dictionaries/phoneme_dict.json"
    phoneme_dict = PhonemeDictionary(lang="en-us", vocab_path=dict_path)

    # Initialize utilities
    embedding_extractor = EmbeddingExtractor()
    mel_extractor = MelExtractor()

    # Initialize dataloader
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

    # Initialize model
    model = ETTSModel(
        num_phonemes=phoneme_dict.get_num_phonemes(), upsample_factor=upsample_factor
    )
    model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()  # for mel spectrogram regression
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (
            phonemes,
            phoneme_lengths,
            speaker_embs,
            mels,
            mel_lengths,
        ) in enumerate(dataloader):
            phonemes = phonemes.to(device)
            speaker_emb = speaker_embs.to(device)
            mels = mels.to(device)

            optimizer.zero_grad()
            outputs = model(phonemes, speaker_emb)
            loss = criterion(outputs, mels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader):.4f}")

    # Save the model checkpoint
    torch.save(model.state_dict(), "etts_model.pth")
    print("Model saved!")


if __name__ == "__main__":
    train()
