import json
import torch
from torch.utils.data import Dataset
from phoneme_dictionary import PhonemeDictionary
from embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor


class ETTSDataset(Dataset):
    """
    Custom PyTorch Dataset for loading phoneme sequences, speaker embeddings, and mel spectrograms
    from a manifest JSON file describing your training samples.
    """

    def __init__(
        self,
        manifest_path: str,
        phoneme_dict: PhonemeDictionary,
        embedding_extractor: EmbeddingExtractor,
        mel_extractor: MelExtractor,
        max_phonemes: int = 256,
        max_mel_length: int = 1024,
    ):
        # Load manifest file (JSON list of {transcript, audio_path, language})
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        self.phoneme_dict = phoneme_dict
        self.embedding_extractor = embedding_extractor
        self.mel_extractor = mel_extractor
        self.max_phonemes = max_phonemes
        self.max_mel_length = max_mel_length

    def __len__(self):
        # Total number of samples in the dataset
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Returns one training example:
        - phoneme sequence as tensor of indices
        - speaker embedding vector
        - mel spectrogram as tensor
        """
        entry = self.entries[idx]
        transcript_path = entry["transcript"]
        audio_path = entry["audio"]

        # Read transcript text from file
        transcript_text = self._load_transcript(transcript_path)

        # Update phoneme dictionary with new transcript
        self.phoneme_dict.add_from_transcript(transcript_path)

        # Conver transcript text to phoneme sequence string
        phoneme_seq = self.phoneme_dict.get_phoneme_seq(transcript_text)
        phoneme_ids = [
            self.phoneme_dict.phonemes[ph]["index"]
            for ph in phoneme_seq.split()
            if ph in self.phoneme_dict.phonemes
        ]
        phoneme_ids = phoneme_ids[: self.max_phonemes]  # Truncate if too long
        self.phoneme_dict.save()  # Save updated phoneme dictionary

        # -- Speaker embedding extraction --
        speaker_embedding = self.embedding_extractor.get_embedding(audio_path)

        # -- Mel spectrogram extraction --
        mel = self.mel_extractor.extract_mel_spectrogram(audio_path)
        mel = mel[:, : self.max_mel_length]  # Truncate to max time

        # -- Create tensors --
        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)
        speaker_tensor = torch.tensor(speaker_embedding, dtype=torch.float32)
        mel_tensor = torch.nn.functional.pad(
            mel, (0, self.max_mel_length - mel.shape[1]), mode="constant", value=0.0
        )

        return phoneme_tensor, speaker_tensor, mel_tensor

    def _load_transcript(self, path: str) -> str:
        # Helper to read transcript text from file
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
