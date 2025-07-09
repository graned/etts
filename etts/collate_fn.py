import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Batch is a list of dicts:
    {
        "phonemes": LongTensor [seq_len],
        "speaker_embedding": FloatTensor [256],
        "mel": FloatTensor [80, time_steps]
    }
    """
    phonemes = [item["phonemes"] for item in batch]
    speaker_embeddings = [item["speaker_embedding"] for item in batch]
    mels = [item["mel"].transpose(0, 1) for item in batch]  # [time, n_mels] for padding

    # Pad phoneme sequences (padding value = 0 for <pad>)
    phonemes_padded = pad_sequence(phonemes, batch_first=True, padding_value=0)

    # Pad mel spectrograms on time dimension
    mels_padded = pad_sequence(mels, batch_first=True)  # pads time dim

    # Transpose mel back to [batch, n_mels, time]
    mels_padded = mels_padded.transpose(1, 2)

    speaker_embeddings = torch.stack(speaker_embeddings)

    return {
        "phonemes": phonemes_padded,  # [batch, seq_len]
        "speaker_embedding": speaker_embeddings,  # [batch, 256]
        "mel": mels_padded,  # [batch, 80, max_time]
    }
