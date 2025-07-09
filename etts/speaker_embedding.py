from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import torch
import os


def get_speaker_embedding(audio_path: str) -> np.ndarray:
    """
    Load an audio file and extract its speaker embedding vector using Resemblyzer.

    Args:
        audio_path (str): Path to the audio file (wav, mp3, etc.)

    Returns:
        np.ndarray: 1D numpy array with the speaker embedding vector (usually 256 dims)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and preprocess the audio to waveform
    wav = preprocess_wav(audio_path)

    # Create the encoder (loads pretrained weights automatically)
    encoder = VoiceEncoder()

    # Get the embedding vector
    embedding = encoder.embed_utterance(wav)

    return embedding


if __name__ == "__main__":
    # Example usage:
    test_audio = "data/test.wav"  # replace with your file path
    embedding = get_speaker_embedding(test_audio)
    print(f"Speaker embedding shape: {embedding.shape}")
    print(embedding)
