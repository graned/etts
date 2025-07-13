# 🗣️ ETTS – Echora Text-to-Speech Engine

**ETTS** is a custom deep learning pipeline for generating expressive speech using PyTorch.  
It is designed to support multilingual, multi-speaker voice synthesis — including future capabilities like prompt-to-voice and voice cloning.

---

## Project Structure

```
etts
├── data
│   ├── etts_dataloader.py
│   └── etts_dataset.py
│  
├── models
│   └── etts_model.py
├── pyrightconfig.json
├── README.md
├── requirements.txt
├── test
│   ├── dummy_dictionaries
│   │   └── phoneme_dict.json
│   ├── dummy_samples
│   ├── embedding_extractor.test.py
│   ├── etts_dataloader.test.py
│   ├── etts_dataset.test.py
│   ├── etts_model.test.py
│   ├── manifest_builder.test.py
│   ├── mel_extractor.test.py
│   ├── outputs
│   │   └── dummy_manifest.json
│   └── phoneme_dictionary.test.py
├── train
│   ├── checkpoints
│   ├── dictionaries
│   │   └── phoneme_dict.json
│   ├── manifests
│   │   └── etts_manifest.json
│   ├── samples
│   │   └── en_us
│   │       └── sample1
│   │           ├── audio.mp3
│   │           └── transcript.txt
│   └── train_etts.py
└── utils
    ├── embedding_extractor.py
    ├── manifest_builder.py
    ├── mel_extractor.py
    └── phoneme_dictionary.py


```

---

## 💡 Conceptual Overview

Human speech is the result of three critical components:

1. **Phonemes** — The basic units of sound (e.g., "k", "aa", "t" in "cat").
2. **Speaker Embedding** — A learned vector that captures a person's unique voice characteristics.
3. **Mel Spectrogram** — A time-frequency representation of audio, showing how pitch and energy evolve over time.

These pieces are combined in a neural network to generate speech from text input.

---

## 🧠 How It Works

     ┌────────────┐       ┌──────────────┐      ┌────────────┐
     │  Text      │──────▶│  Phonemizer  │────▶ │ Phoneme IDs│
     └────────────┘       └──────────────┘      └────────────┘
                                                        │
                                                        ▼
                                       ┌─────────────────────────┐
                                       │  Speaker Embedding (SE) │
                                       └─────────────────────────┘
                                                        │
                                                        ▼
                                     ┌────────────────────────────┐
                                     │       ETTS Model           │
                                     │ (Phonemes + SE → Mel Spec) │
                                     └────────────────────────────┘
                                                        │
                                                        ▼
                               ┌────────────────────────────────────────┐
                               │      Vocoder (HiFi-GAN or similar)     │
                               └────────────────────────────────────────┘
                                                        │
                                                        ▼
                                              🔊 Final Audio Output

---

## 🛠️ Components

| Module         | Description                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `model.py`     | PyTorch model that takes phoneme sequences and speaker embeddings, and predicts mel spectrograms |
| `train.py`     | Basic training loop using synthetic or real data                                                 |
| `dataset.py`   | Custom Dataset class that loads `.npz` files with phonemes, embeddings, and mel spectrograms     |
| `utils.py`     | Helper functions for phonemizing, visualizing, and processing audio                              |
| `scripts/`     | Tools for preprocessing audio, generating dummy data, and more                                   |
| `checkpoints/` | Where trained model weights will be saved                                                        |

---

## 📦 Dataset Format

Training data is stored in `.npz` files with the following keys:

- `phonemes`: (T,) — array of phoneme IDs
- `embedding`: (256,) — speaker embedding
- `mel`: (T, 80) — mel spectrogram with 80 bins

---

## 🔁 Training Loop (Simplified)

1. Load phoneme IDs
2. Load speaker embedding
3. Load ground-truth mel spectrogram
4. Predict mel spectrogram from model
5. Compute loss (MSE)
6. Backpropagation and optimizer step

---

## 📈 Visual: Mel Spectrogram

Here's a mel spectrogram of the word **"hello"**:

![mel spectrogram example](https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Spectrogram-19thC.png/800px-Spectrogram-19thC.png)

> The horizontal axis is time; the vertical is frequency; color intensity shows energy.

---

## 🧪 Goals and Future Ideas

- [x] Modular pipeline for phoneme + embedding → mel
- [ ] Add HiFi-GAN vocoder for final waveform synthesis
- [ ] Voice cloning from reference audio
- [ ] Prompt-to-voice generation using descriptions (e.g., "older female, calm, storyteller")
- [ ] Multi-language support via `phonemizer` and language-specific tokenizers
- [ ] TinyML export or serverless inference (low-latency TTS)

---

## ⚙️ Installation

```bash
pip install -r requirements.txt

```
