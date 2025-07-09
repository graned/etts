import torch
from torch.utils.data import Dataset
from typing import List, Dict
from phoneme_converter import text_to_phonemes
from speaker_embedding import get_speaker_embedding
from mel_generator import audio_file_to_mel
from phoneme_vocab import phoneme_to_sequence


class TTSDataSet(Dataset):
    def __init__(self, data: List[Dict], phoneme_lang: str = "en-us"):
        """
        Args:
            data: List of dicts with keys: 'audio_path', 'transcript', and optional 'language'
            phoneme_lang: default language for phonemizer (overridden by per-item 'language' if present)
        """
        self.data = data
        self.default_lang = phoneme_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract phoneme IDs
        lang = item.get("language", self.default_lang)
        phoneme_str = text_to_phonemes(item["transcript"], lang)
        phoneme_ids = phoneme_to_sequence(phoneme_str)

        # Get speaker embedding
        speaker_embedding = get_speaker_embedding(item["audio_path"])

        # Get mel spectrogram
        mel = audio_file_to_mel(item["audio_path"])

        return {
            "phonemes": torch.tensor(phoneme_ids, dtype=torch.long),
            "speaker_embedding": torch.tensor(speaker_embedding, dtype=torch.float32),
            "mel": mel,
        }
