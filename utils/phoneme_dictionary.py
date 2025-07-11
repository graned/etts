import json
import os
from phonemizer import phonemize
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

    def save(self):
        data = {
            "phonemes": self.phonemes,
            "metadata": {
                "language": self.lang,
                "note": "Phoneme dictionary with numeric indices and reference mel placeholders",
                # "phoneme_coverage_percent": self.coverage(),
            },
        }
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved phoneme dictionary to {self.vocab_path}")

    def encode_text(self, text: str) -> list[int]:
        phoneme_seq = phonemize(
            text,
            language=self.lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
        )
        phoneme_seq = cast(str, phoneme_seq)
        return [
            self.phonemes[p]["index"] for p in phoneme_seq.split() if p in self.phonemes
        ]

    def get_phoneme_seq(self, text: str) -> str:
        phoneme_seq = phonemize(
            text,
            language=self.lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
        )
        return cast(str, phoneme_seq)

    def add_from_transcript(self, transcript_path: str):
        # Read transcript text
        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Generate phonemes using phonemizer
        phoneme_seq = phonemize(
            text,
            language=self.lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
        )
        phoneme_seq = cast(str, phoneme_seq)  # Ensure it's a string
        # phoneme_seq is a string of phonemes separated by spaces, e.g. "h ə l oʊ"

        for phoneme in phoneme_seq.split():
            if phoneme not in self.phonemes:
                self.phonemes[phoneme] = {
                    "index": self.next_index,
                    "symbol": phoneme,
                }
                print(f"➕ Added new phoneme '{phoneme}' with index {self.next_index}")
                self.next_index += 1
            else:
                # phoneme already exists, skip
                pass
