import json
import torch
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.etts_model import ETTSModel


def main():
    vocab_path = (
        "test/dummy_dictionaries/phoneme_dict.json"  # Update with your vocab path
    )

    # Load phoneme dictionary
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phonemes_dict = data.get("phonemes", {})
    num_phonemes = len(phonemes_dict)
    print(f"Loaded phoneme dictionary with {num_phonemes} phonemes")

    # Create ETTS model with dynamic phoneme vocab size
    model = ETTSModel(num_phonemes=num_phonemes)

    # Create dummy inputs (batch size 1)
    dummy_phonemes = torch.randint(0, num_phonemes, (1, 10))  # sequence length 10
    dummy_speaker_emb = torch.randn(1, 256)  # embedding dim 256

    # Forward pass
    mel_output = model(dummy_phonemes, dummy_speaker_emb)

    print(f"Output mel spectrogram shape: {mel_output.shape}")


if __name__ == "__main__":
    main()
