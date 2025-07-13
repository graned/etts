import json
import os
from re import split
from phonemizer import phonemize
from phonemizer.separator import Separator
from typing import cast


class PhonemeDictionary:
    def __init__(self, lang: str, vocab_path: str, expected_phonemes=set):
        self.lang = lang
        self.vocab_path = vocab_path
        self.phonemes = {}  # { "symbol": str, "id"": int, "ref_mel": [] }
        self.next_index = 1  # reserved 0 for padding
        self.expected_phonemes = expected_phonemes

        # Load vocabulary if available
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.phonemes = data.get("phonemes", {})
                if self.phonemes:
                    self.next_index = (
                        max(p["index"] for p in self.phonemes.values()) + 1
                    )

    def get_num_phonemes(self) -> int:
        """
        Returns the number of phonemes in the dictionary.
        Includes padding (index 0), so this is safe to use for nn.Embedding.
        """
        return max(p["index"] for p in self.phonemes.values()) + 1

    def save(self):
        data = {
            "phonemes": self.phonemes,
            "metadata": {
                "language": self.lang,
                "note": "Phoneme dictionary with numeric indices and reference mel placeholders",
            },
        }
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    #        print(f"💾 Saved phoneme dictionary to {self.vocab_path}")

    def encode_text(self, text: str) -> list[int]:
        phoneme_seq = self.get_phoneme_seq(text)
        return [
            self.phonemes[p]["index"] for p in phoneme_seq.split() if p in self.phonemes
        ]

    def get_phoneme_seq(self, text: str) -> str:
        phoneme_seq = phonemize(
            text,
            language=self.lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            separator=Separator(phone="|", word=" "),
        )
        # replace '|' with space and strip whitespace
        phoneme_seq = cast(str, phoneme_seq)
        return phoneme_seq.replace("|", " ").strip()

    def load_from_manifest(self, manifest_path: str):
        """
        Load phonemes from a manifest file.
        The manifest should be a JSON list of {transcript, audio_path, language}.
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            transcript_path = entry["transcript"]
            self.add_from_transcript(transcript_path)

    def add_from_transcript(self, transcript_path: str):
        # Read transcript text
        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Generate phonemes using phonemizer
        phoneme_seq = self.get_phoneme_seq(text)
        # phoneme_seq is a string of phonemes separated by spaces, e.g. "h ə l oʊ"

        for phoneme in phoneme_seq.split():
            if phoneme not in self.phonemes:
                self.phonemes[phoneme] = {
                    "index": self.next_index,
                    "symbol": phoneme,
                }
                self.next_index += 1
                print(f"➕ Added new phoneme '{phoneme}' with index {self.next_index}")
                self.save()
            else:
                # phoneme already exists, skip
                pass
