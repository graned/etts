import librosa
import torch
import numpy as np


class MelExtractor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract_mel_spectrogram(self, audio_path: str) -> torch.Tensor:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=1.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.from_numpy(mel_db).float()  # [n_mels, time]
