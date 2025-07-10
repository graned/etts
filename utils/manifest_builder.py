import os
import json
from glob import glob
from typing import List, Dict


class ManifestBuilder:
    def __init__(self, root_dir: str = "train_samples"):
        self.root_dir = root_dir
        self.entries: List[Dict[str, str]] = []

    def build(self) -> List[Dict]:
        print(f"📂 Building manifest from root directory: {self.root_dir}...")
        """
        Scans the root_dir for samples and builds the manifest.
        Expected structure: root_dir/lang/sample/{*.wav|*.mp3, *.txt}
        """
        self.entries = []

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

                entry = {
                    "transcript": transcription_file[0],
                    "audio": audio_file[0],
                    "language": lang_folder,
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
