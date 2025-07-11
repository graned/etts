import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.phoneme_dictionary import PhonemeDictionary
from utils.embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor
from data.etts_dataloader import ETTSDataloader


def test_dataloader():
    phoneme_dict = PhonemeDictionary(
        "en-us", "test/dummy_dictionaries/phoneme_dict.json"
    )
    embedding_extractor = EmbeddingExtractor()
    mel_extractor = MelExtractor()

    dataloader_builder = ETTSDataloader(
        phoneme_dict=phoneme_dict,
        embedding_extractor=embedding_extractor,
        mel_extractor=mel_extractor,
        batch_size=5,
        shuffle=False,
        num_workers=0,  # Use 0 for simpler debugging
    )

    loader = dataloader_builder.load("test/outputs/dummy_manifest.json")

    for batch in loader:
        phonemes, phoneme_lengths, speaker_embs, mels, mel_lengths = batch
        print("Phonemes shape:", phonemes.shape)
        print("Phoneme lengths:", phoneme_lengths)
        print("Speaker embeddings shape:", speaker_embs.shape)
        print("Mels shape:", mels.shape)
        print("Mel lengths:", mel_lengths)
        break


if __name__ == "__main__":
    test_dataloader()
