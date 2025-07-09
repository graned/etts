from phonemizer import phonemize
import re
import json
from typing import List, Dict


def phonemize_text(text: str, lang: str) -> str:
    """
    Clean and phonemize a single text string.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9' ]+", "", text)  # simple cleaner
    phonemes = phonemize(
        text, language=lang, backend="espeak", strip=True, with_stress=False
    )
    return phonemes.strip()


def extract_unique_phonemes(data: List[Dict], default_lang: str = "en-us") -> List[str]:
    """
    Takes in a dataset and returns a sorted list of unique IPA phonemes.
    """
    phoneme_set = set()
    for item in data:
        lang = item.get("language", default_lang)
        text = item["transcript"]
        phoneme_str = phonemize_text(text, lang)
        phonemes = phoneme_str.split()
        phoneme_set.update(phonemes)

    phoneme_list = sorted(list(phoneme_set))
    return phoneme_list


def save_vocab(phonemes: List[str], out_path: str = "phoneme_vocab.json"):
    """
    Saves phoneme-to-ID and ID-to-phoneme mappings.
    """
    special_tokens = ["<pad>", "<unk>", "sil"]
    vocab = special_tokens + phonemes

    phoneme_to_id = {p: i for i, p in enumerate(vocab)}
    id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

    with open(out_path, "w") as f:
        json.dump(
            {"phoneme_to_id": phoneme_to_id, "id_to_phoneme": id_to_phoneme},
            f,
            indent=2,
        )

    print(f"Saved vocab with {len(vocab)} tokens to {out_path}")


if __name__ == "__main__":
    # Replace this with your real dataset list
    dummy_data = [
        {"transcript": "Hello, how are you?", "language": "en-us"},
        {"transcript": "¿Cómo estás?", "language": "es"},
        {"transcript": "Ich heiße Anna.", "language": "de"},
    ]

    phonemes = extract_unique_phonemes(dummy_data)
    save_vocab(phonemes)
