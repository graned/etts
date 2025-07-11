import torch
import torch.nn as nn


class ETTSModel(nn.Module):
    """
    Echora TTS Model (ETTSModel)

    This model takes a sequence of phoneme IDs and a speaker embedding,
    and predicts a mel spectrogram representing the speech.

    Inputs:
    - phoneme_ids: Tensor of shape (B, T_phonemes)
    - speaker_embeddings: Tensor of shape (B, 256)

    Output:
    - mel spectrogram: Tensor of shape (B, 80, T)
    """

    def __init__(
        self,
        num_phonemes: int,
        phoneme_embedding_dim: int = 256,
        speaker_embedding_dim: int = 256,
        hidden_dim: int = 512,
        mel_dim: int = 80,
        num_layers: int = 2,
    ):
        super().__init__()

        # Learnable phoneme embedding table
        self.phoneme_embedding = nn.Embedding(num_phonemes, phoneme_embedding_dim)

        # Project speaker embedding to match phoneme embedding size
        self.speaker_projection = nn.Linear(
            speaker_embedding_dim, phoneme_embedding_dim
        )

        # Bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=phoneme_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Projection to mel spectrogram
        self.mel_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mel_dim),
        )

    def forward(self, phoneme_ids: torch.Tensor, speaker_embeddings: torch.Tensor):
        """
        Forward pass of the model.

        Returns:
        - mel_spectrogram: Tensor (B, 80, T)
        """
        B, T = phoneme_ids.size()

        # Embed phonemes
        phoneme_embed = self.phoneme_embedding(phoneme_ids)  # (B, T, D)

        # Project speaker embedding and broadcast to time steps
        speaker_proj = self.speaker_projection(speaker_embeddings).unsqueeze(
            1
        )  # (B, 1, D)
        speaker_proj = speaker_proj.expand(-1, T, -1)  # (B, T, D)

        # Combine speaker + phoneme
        x = phoneme_embed + speaker_proj

        # Encode temporal dynamics
        encoded, _ = self.encoder(x)  # (B, T, 2H)

        # Project to mel spectrogram
        mel = self.mel_projection(encoded)  # (B, T, mel_dim)

        return mel.transpose(1, 2)  # (B, mel_dim, T)
