import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.embedding_extractor import EmbeddingExtractor

if __name__ == "__main__":
    embedding_extractor = EmbeddingExtractor()

    # path to audio file
    audio_path = "test/dummy_samples/en_us/sample1/audio.mp3"
    embedding = embedding_extractor.get_embedding(audio_path)

    print(f"Got embedding of shape {embedding.shape} for audio {audio_path}")
