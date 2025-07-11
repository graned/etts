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
        phoneme_ids = []

        # Get mel spectrogram for the audio
        mel = self.mel_extractor.extract_mel_spectrogram(audio_path)
        for ph in phoneme_seq.split():
            if ph not in self.phoneme_dict.phonemes:
                print(
                    f"⚠️  Unexpected: phoneme '{ph}' not in dictionary even after add_from_transcript"
                )
                continue

            entry = self.phoneme_dict.phonemes[ph]
            phoneme_ids.append(entry["index"])

            # Initialize mel spectrogram placeholder if not present
            if not entry["reference_mel"]:
                # Crude init: assign the first N frames from mel
                init_frames = (
                    mel[:, : self.max_mel_length].cpu().numpy().tolist()
                )  # Get first 10 frames
                entry["reference_mel"] = init_frames
                print(f"🧠 Initialized reference mel for phoneme '{ph}'")

        self.phoneme_dict.save()  # Save updated phoneme dictionary

        phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.long)

        # Extract speaker embedding from the audio
        speaker_embedding = self.embedding_extractor.get_embedding(audio_path)
        speaker_tensor = torch.tensor(speaker_embedding, dtype=torch.float32)

        mel = mel[:, : self.max_mel_length]  # Truncate to max time
        mel_tensor = torch.nn.functional.pad(
            mel, (0, self.max_mel_length - mel.shape[1]), mode="constant", value=0.0
        )

        return phoneme_tensor, speaker_tensor, mel_tensor

    def _load_transcript(self, path: str) -> str:
        # Helper to read transcript text from file
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
