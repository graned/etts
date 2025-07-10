import librosa
import numpy as np
import os
import torch


def load_audio(audio_path: str, sample_rate: int = 22050) -> np.ndarray:
    """
    Loads an audio file and resamples to the target sample rate.

    Args:
        audio_path (str): Path to audio file.
        sample_rate (int): Target sample rate (default: 22050).

    Returns:
        np.ndarray: Audio time series.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return audio


def audio_to_mel(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: int = 0,
    fmax: int = 8000,
) -> np.ndarray:
    """
    Converts a waveform to a mel spectrogram.

    Returns:
        np.ndarray: (n_mels, time) mel spectrogram.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db


def audio_file_to_mel(audio_path: str) -> torch.Tensor:
    """
    Complete pipeline: load audio and convert to mel spectrogram.

    Args:
        audio_path (str): Path to audio file.

    Returns:
        torch.Tensor: Mel spectrogram tensor of shape [n_mels, T]
    """
    audio = load_audio(audio_path)
    mel = audio_to_mel(audio)
    mel_tensor = torch.from_numpy(mel).float()
    return mel_tensor


if __name__ == "__main__":
    # Example usage
    test_audio = "data/test.wav"
    mel = audio_file_to_mel(test_audio)
    print("Mel spectrogram shape:", mel.shape)  # [n_mels, time_steps]
