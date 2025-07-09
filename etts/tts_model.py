import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even idx
        pe[:, 1::2] = torch.cos(position * div_term)  # odd idx
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: [seq_len, batch, d_model]
        attn_output, _ = self.self_attn(
            x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x


class PostNet(nn.Module):
    def __init__(self, n_mels=80, n_convolutions=5, channels=512, kernel_size=5):
        super().__init__()
        layers = []
        for i in range(n_convolutions):
            in_channels = n_mels if i == 0 else channels
            out_channels = n_mels if i == n_convolutions - 1 else channels
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size, padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.Tanh() if i != n_convolutions - 1 else nn.Identity(),
                    nn.Dropout(0.5),
                )
            )
        self.convs = nn.ModuleList(layers)

    def forward(self, x):
        # x: [batch, n_mels, seq_len]
        for conv in self.convs:
            x = conv(x)
        return x


class TTSModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        speaker_emb_dim=256,
        n_mels=80,
        max_seq_len=1000,
        dropout=0.1,
    ):
        super().__init__()

        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = TransformerBlock(d_model, nhead, dim_feedforward, dropout)
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

        # Speaker embedding projection and broadcast
        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)

        # Final linear to mel dimension
        self.mel_linear = nn.Linear(d_model, n_mels)

        # Post-net for refinement
        self.postnet = PostNet(n_mels=n_mels)

    def forward(self, phoneme_seq, speaker_emb, phoneme_mask=None):
        """
        Args:
            phoneme_seq: [batch, seq_len] LongTensor
            speaker_emb: [batch, speaker_emb_dim] FloatTensor
            phoneme_mask: [batch, seq_len] BoolTensor (True for pad tokens)
        Returns:
            mel_pred: [batch, n_mels, seq_len]
            mel_postnet: [batch, n_mels, seq_len]
        """
        x = self.phoneme_embedding(phoneme_seq)  # [batch, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch, seq_len, d_model]

        # Transformer expects [seq_len, batch, d_model]
        x = x.transpose(0, 1)

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=phoneme_mask)

        x = x.transpose(0, 1)  # [batch, seq_len, d_model]

        # Add speaker conditioning
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)  # [batch, 1, d_model]
        x = x + spk  # broadcast add

        mel_pred = self.mel_linear(x)  # [batch, seq_len, n_mels]

        # Transpose to [batch, n_mels, seq_len] for postnet
        mel_pred = mel_pred.transpose(1, 2)

        mel_postnet = mel_pred + self.postnet(mel_pred)  # residual connection

        return mel_pred, mel_postnet
