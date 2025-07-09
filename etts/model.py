# etts/model.py
import torch.nn as nn


class ETTSTransformer(nn.Module):
    def __init__(self, num_phonemes=100, embedding_dim=256, mel_bins=80):
        super().__init__()
        self.embedding = nn.Embedding(num_phonemes, 128)
        self.lstm = nn.LSTM(128 + embedding_dim, 256, batch_first=True)
        self.linear = nn.Linear(256, mel_bins)

    def forward(self, phoneme_ids, speaker_embedding):
        B, T = phoneme_ids.shape
        x = self.embedding(phoneme_ids)  # (B, T, 128)
        speaker_exp = speaker_embedding.unsqueeze(1).repeat(1, T, 1)  # (B, T, 256)
        x = torch.cat([x, speaker_exp], dim=-1)  # (B, T, 384)
        out, _ = self.lstm(x)
        return self.linear(out)  # (B, T, mel_bins)
