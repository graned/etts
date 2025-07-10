import os
import sys
from torch.utils.data import DataLoader

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phoneme_dictionary import PhonemeDictionary
from embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor
from train.etts_dataset import ETTSDataset


# Paths
manifest_path = "test/outputs/dummy_manifest.json"  # update if needed
vocab_path = "test/dummy_dictionaries/phoneme_dict.json"  # update if needed
lang = "en-us"

# Load supporting tools
phoneme_dict = PhonemeDictionary(lang=lang, vocab_path=vocab_path)
embedding_extractor = EmbeddingExtractor()
mel_extractor = MelExtractor()

# Create dataset and dataloader
dataset = ETTSDataset(manifest_path, phoneme_dict, embedding_extractor, mel_extractor)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate over one batch
for phonemes, speaker_embeddings, mels in loader:
    print("Phoneme IDs:", phonemes.shape)  # [batch, time]
    print("Speaker embeddings:", speaker_embeddings.shape)  # [batch, 256]
    print("Mel spectrograms:", mels.shape)  # [batch, 80, max_len]
    break
