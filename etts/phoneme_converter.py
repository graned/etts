from phonemizer import phonemize
import re


def clean_text(text: str) -> str:
    """
    Basic text cleanup: lowercase, strip punctuation except apostrophes.
    """
    text = text.lower()
    # Keep letters, digits, apostrophes and spaces
    text = re.sub(r"[^a-z0-9' ]+", "", text)
    return text.strip()


def text_to_phonemes(text: str, lang: str = "en-us") -> str:
    """
    Convert input text to phonemes using phonemizer.

    Args:
        text (str): Raw input transcript.
        lang (str): Language code for phonemizer (e.g. 'en-us', 'de', 'es').

    Returns:
        str: Space-separated phoneme sequence.
    """
    cleaned = clean_text(text)
    phonemes = phonemize(
        cleaned,
        language=lang,
        backend="espeak",
        strip=True,
        with_stress=False,  # Change to True if you want stress marks
        njobs=1,
    )
    return phonemes


if __name__ == "__main__":
    # Example usage
    example_text = "Hello, how are you doing today?"
    print("Input text:", example_text)
    print("Phonemes:", text_to_phonemes(example_text, lang="en-us"))
