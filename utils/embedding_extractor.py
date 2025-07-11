import numpy as np
import librosa
from resemblyzer import VoiceEncoder
from typing import cast


class EmbeddingExtractor:
    def __init__(self):
        self.encoder = VoiceEncoder()

    def get_embedding(self, audio_path: str) -> np.ndarray:
        # librosa loads audio as float32, mono by default, at sample rate 22050 by default
        wav, sr = librosa.load(audio_path, sr=16000)  # resample to 16kHz

        # Resemblyzer expects 16kHz mono float32 normalized audio
        embedding = cast(np.ndarray, self.encoder.embed_utterance(wav))
        return embedding
