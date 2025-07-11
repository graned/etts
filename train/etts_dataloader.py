import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from train.etts_dataset import ETTSDataset
from phoneme_dictionary import PhonemeDictionary
from embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor


class ETTSDataloader:
    def __init__(
        self,
        phoneme_dict: PhonemeDictionary,
        embedding_extractor: EmbeddingExtractor,
        mel_extractor: MelExtractor,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 2,
    ):
        self.phoneme_dict = phoneme_dict
        self.embedding_extractor = embedding_extractor
        self.mel_extractor = mel_extractor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def _collate_fn(self, batch):
        """
        Collate function to pad phoneme sequences and mel spectrograms.
        Returns padded tensors and length arrays.
        """
        phoneme_seqs, speaker_embs, mel_specs = zip(*batch)

        phoneme_lengths = torch.tensor([len(seq) for seq in phoneme_seqs])
        padded_phonemes = pad_sequence(phoneme_seqs, batch_first=True, padding_value=0)

        mel_lengths = torch.tensor([mel.shape[1] for mel in mel_specs])
        max_mel_len = mel_lengths.max().item()
        padded_mels = torch.stack(
            [
                torch.nn.functional.pad(
                    mel, (0, max_mel_len - mel.shape[1]), mode="constant", value=0.0
                )
                for mel in mel_specs
            ]
        )

        speaker_embs = torch.stack(speaker_embs)

        return padded_phonemes, phoneme_lengths, speaker_embs, padded_mels, mel_lengths

    def load(self, manifest_path: str) -> DataLoader:
        """
        Creates a DataLoader for the dataset at the given manifest path.
        """
        dataset = ETTSDataset(
            manifest_path=manifest_path,
            phoneme_dict=self.phoneme_dict,
            embedding_extractor=self.embedding_extractor,
            mel_extractor=self.mel_extractor,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
