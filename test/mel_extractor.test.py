import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.mel_extractor import MelExtractor

if __name__ == "__main__":
    mel_extractor = MelExtractor()

    # path to audio file
    audio_path = "test/dummy_samples/en_us/sample1/audio.mp3"
    mel = mel_extractor.extract_mel_spectrogram(audio_path)

    print(f"Got mel spectrogram of shape {mel.shape} for audio {audio_path}")
