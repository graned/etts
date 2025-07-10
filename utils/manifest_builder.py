import os
import json
import hashlib
from glob import glob
from typing import List, Dict, Optional


class ManifestBuilder:
    def __init__(self, root_dir: str = "train_samples"):
        self.root_dir = root_dir
        self.entries: List[Dict[str, str]] = []

    def _hash_audio(self, audio_path: str, chunk_size: int = 8192) -> Optional[str]:
        """
        Compute SHA256 hash of the audio file to detect duplicates.
        """
        try:
            hasher = hashlib.sha256()
            with open(audio_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"❌ Failed hashing audio {audio_path}: {e}")
            return None

    """
    Scans the root_dir for samples and builds the manifest.
    Expected structure: root_dir/lang/sample/{*.wav|*.mp3, *.txt}
    Args:
        deduplicate: If True, skip audio files with duplicate hashes.
    """

    def build(self, deduplicate: bool = True) -> List[Dict]:
        print(f"📂 Building manifest from root directory: {self.root_dir}...")

        self.entries = []
        seen_hashes = set()

        for lang_folder in os.listdir(self.root_dir):
            lang_path = os.path.join(self.root_dir, lang_folder)
            print(f"🔍 Scanning language folder: {lang_path}")
            if not os.path.isdir(lang_path):
                continue

            for sample_folder in os.listdir(lang_path):
                print(f"🔍 Scanning sample folder: {sample_folder}")
                sample_path = os.path.join(lang_path, sample_folder)
                print(f"📂 Processing sample: {sample_path}")
                if not os.path.isdir(sample_path):
                    continue

                transcription_file = glob(os.path.join(sample_path, "*.txt"))
                audio_file = glob(os.path.join(sample_path, "*.wav")) + glob(
                    os.path.join(sample_path, "*.mp3")
                )

                if not transcription_file:
                    print(f"⚠️ Missing transcript in {sample_path}")
                    continue
                if not audio_file:
                    print(f"⚠️ Missing audio in {sample_path}")
                    continue

                audio_path = audio_file[0]

                if deduplicate:
                    audio_hash = self._hash_audio(audio_path)
                    if audio_hash is None:
                        continue  # skip if hash failed
                    if audio_hash in seen_hashes:
                        print(f"⚠️ Duplicate audio detected, skipping {audio_path}")
                        continue
                    seen_hashes.add(audio_hash)

                entry = {
                    "transcript": transcription_file[0],
                    "audio": audio_path,
                    "language": lang_folder,
                    # TODO: Add metadata fields like duration, sample rate here
                    # TODO: Validate audio format, sample rate, and channels
                    # TODO: Support filtering by duration (min/max)
                    # TODO: Support automatic conversion of mp3 to wav if needed
                }

                self.entries.append(entry)

        print(f"✅ Built manifest with {len(self.entries)} samples")
        return self.entries

    def save(self, path: str = "data/manifest.json"):
        if not self.entries:
            raise RuntimeError("Manifest is empty. Please build it first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        print(f"💾 Manifest saved to {path}")

    def load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        print(f"📂 Loaded {len(self.entries)} samples from {path}")
        return self.entries
