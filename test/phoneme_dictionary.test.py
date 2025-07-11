import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.phoneme_dictionary import PhonemeDictionary

if __name__ == "__main__":
    # vocab directory
    path = "test/dummy_dictionaries"
    phoneme_dict = PhonemeDictionary(
        lang="en-us", vocab_path=os.path.join(path, "phoneme_dict.json")
    )

    # path to transcript file
    transcript_path = "test/dummy_samples/en_us/sample1/transcript.txt"
    phoneme_dict.add_from_transcript(transcript_path)

    phoneme_dict.save()
    print("Dictionary file saved")
