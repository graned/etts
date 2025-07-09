# 🗣️ ETTS – Echora Text-to-Speech Engine

**ETTS** is a custom deep learning pipeline for generating expressive speech using PyTorch.  
It is designed to support multilingual, multi-speaker voice synthesis — including future capabilities like prompt-to-voice and voice cloning.

---

## Project Structure

```
etts/
├── etts/                    ← Python package
│   ├── __init__.py
│   ├── model.py             ← Your PyTorch model definition
│   ├── train.py             ← Training loop
│   ├── dataset.py           ← Custom dataset loader
│   ├── utils.py             ← Helpers (phonemizer, spectrograms, etc.)
│
├── scripts/                 ← Optional tools for audio/text preprocessing
│   └── create_dummy_data.py
│
├── checkpoints/             ← Saved model weights
│
├── requirements.txt
├── README.md
└── main.py                  ← CLI entrypoint (e.g., `python main.py --infer "Hello"`)
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

| Module | Description |
|--------|-------------|
| `model.py` | PyTorch model that takes phoneme sequences and speaker embeddings, and predicts mel spectrograms |
| `train.py` | Basic training loop using synthetic or real data |
| `dataset.py` | Custom Dataset class that loads `.npz` files with phonemes, embeddings, and mel spectrograms |
| `utils.py` | Helper functions for phonemizing, visualizing, and processing audio |
| `scripts/` | Tools for preprocessing audio, generating dummy data, and more |
| `checkpoints/` | Where trained model weights will be saved |

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

