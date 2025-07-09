import os
import numpy as np

os.makedirs("data/train", exist_ok=True)

for i in range(100):
    phonemes = np.random.randint(0, 100, (50,))
    embedding = np.random.rand(256).astype(np.float32)
    mel = np.random.rand(50, 80).astype(np.float32)
    np.savez(
        f"data/train/sample_{i}.npz", phonemes=phonemes, embedding=embedding, mel=mel
    )
