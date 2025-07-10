import json
import os

# Load vocab at import
_VOCAB_PATH = "phoneme_vocab.json"

if not os.path.exists(_VOCAB_PATH):
    raise FileNotFoundError(
        f"Phoneme vocab file not found at {_VOCAB_PATH}. "
        "Run build_phoneme_vocab.py first."
    )

with open(_VOCAB_PATH, "r") as f:
    vocab_data = json.load(f)

PHONEME_TO_ID = vocab_data["phoneme_to_id"]
ID_TO_PHONEME = {int(k): v for k, v in vocab_data["id_to_phoneme"].items()}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SIL_TOKEN = "sil"

PAD_ID = PHONEME_TO_ID[PAD_TOKEN]
UNK_ID = PHONEME_TO_ID[UNK_TOKEN]
SIL_ID = PHONEME_TO_ID[SIL_TOKEN]


def phoneme_to_sequence(phoneme_string: str) -> list:
    """
    Converts a space-separated phoneme string to a list of IDs.
    """
    phonemes = phoneme_string.strip().split()
    return [PHONEME_TO_ID.get(p, UNK_ID) for p in phonemes]


def sequence_to_phoneme(seq: list) -> str:
    """
    Converts a list of phoneme IDs back to a string.
    """
    return " ".join([ID_TO_PHONEME.get(i, UNK_TOKEN) for i in seq])


def get_vocab_size() -> int:
    return len(PHONEME_TO_ID)
