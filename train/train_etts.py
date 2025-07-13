import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from models.etts_model import ETTSModel
from data.etts_dataloader import ETTSDataloader
from utils.phoneme_dictionary import PhonemeDictionary
from utils.embedding_extractor import EmbeddingExtractor
from utils.mel_extractor import MelExtractor
from utils.manifest_builder import ManifestBuilder

# 📦 Load components
dict_path = os.getenv("PHONEME_DICT_PATH", "train/dictionaries/phoneme_dict.json")
manifest_path = os.getenv("MANIFEST_PATH", "train/manifests/etts_manifest.json")
ckpt_dir = os.getenv("CHECKPOINT_DIR", "train/checkpoints")
samples_path = os.getenv("SAMPLES_PATH", "train/samples")
t_epochs = int(os.getenv("TRAIN_EPOCHS", 100))
t_lr = float(os.getenv("TRAIN_LR", 1e-3))
p_steps = int(os.getenv("TRAIN_PHONEME_STEPS", 1))
p_mels = int(os.getenv("TRAIN_PHONEME_MELS", 2))
t_lang = os.getenv("TRAIN_LANGUAGE", "en-us")
t_setup_only = os.getenv("TRAIN_SETUP_ONLY", "false").lower() == "true"
t_train_only = os.getenv("TRAIN_ONLY", "false").lower() == "true"
t_tmp_dir = os.getenv("TMPDIR", "tmp")

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(t_tmp_dir, exist_ok=True)

os.environ["TMPDIR"] = t_tmp_dir
phoneme_dict = PhonemeDictionary(lang=t_lang, vocab_path=dict_path)


def setup():
    # Step 1: Build Manifest
    print("🔍 Building manifest...")
    manifest_builder = ManifestBuilder(samples_path)
    manifest_builder.build()
    manifest_builder.save(manifest_path)
    print(f"🔍 Manifest saved to {manifest_path}")

    # 🧠 Step 2: Load Phoneme Dictionary from manifest
    print("📖 Indexing phoneme dictionary...")
    phoneme_dict.load_from_manifest(manifest_path)
    print(
        f"📖 Phoneme dictionary loaded with {phoneme_dict.get_num_phonemes()} phonemes."
    )


def train():
    writer = SummaryWriter(log_dir="runs/etts_experiment")  # TensorBoard writer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize utilities
    embedding_extractor = EmbeddingExtractor()
    mel_extractor = MelExtractor()

    # 🧬 Load data + preprocessing tools
    dataloader_builder = ETTSDataloader(
        phoneme_dict=phoneme_dict,
        embedding_extractor=embedding_extractor,
        mel_extractor=mel_extractor,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Use 0 for simpler debugging
    )
    dataloader = dataloader_builder.load(manifest_path)

    # Get one batch to estimate upsample factor
    phonemes, phoneme_lengths, speaker_embs, mels, mel_lengths = next(iter(dataloader))

    avg_phoneme_len = phonemes.shape[p_steps]  # number of phoneme steps
    avg_mel_len = mels.shape[p_mels]  # number of mel time steps

    upsample_factor = round(avg_mel_len / avg_phoneme_len)
    print(f"Estimated upsample factor: {upsample_factor}")

    # 🏗️ Model, loss, optimizer
    model = ETTSModel(
        num_phonemes=phoneme_dict.get_num_phonemes(), upsample_factor=upsample_factor
    )
    model.to(device)
    # Loss function and optimizer
    criterion = nn.MSELoss()  # for mel spectrogram regression
    optimizer = optim.Adam(model.parameters(), lr=t_lr)
    best_loss = float("inf")

    for epoch in range(t_epochs):
        print(f"\n🌱 Epoch {epoch + 1}/{t_epochs}")
        model.train()
        total_loss = 0.0

        outputs = None
        for batch_idx, (
            phonemes,
            phoneme_lengths,
            speaker_embs,
            mels,
            mel_lengths,
        ) in enumerate(dataloader):
            phonemes = phonemes.to(device)
            speaker_embs = speaker_embs.to(device)
            mels = mels.to(device)

            optimizer.zero_grad()
            outputs = model(phonemes, speaker_embs)
            loss = criterion(outputs, mels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(
                        f"GradientsHist/{name}", param.grad.norm(), batch_idx
                    )

            optimizer.step()

            for group in optimizer.param_groups:
                writer.add_scalar("Learning Rate", group["lr"], batch_idx)

            total_loss += loss.item()

            writer.add_scalar("Loss/train", loss.item(), batch_idx)
            # Log every few steps
            if batch_idx % 10 == 0:
                visualize_mel(writer, mels[0], batch_idx, "Mel/Target")
                visualize_mel(writer, outputs[0], batch_idx, "Mel/Generated")

        avg_loss = total_loss / len(dataloader)
        print(f"📉 Average Loss: {avg_loss:.6f}")

        # 💾 Save checkpoint
        model_path = os.path.join(ckpt_dir, f"etts_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved checkpoint: {model_path}")

        # 🧪 Optional: save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
    writer.close()


def visualize_mel(writer, mel_tensor, step, tag="mel_spectrogram"):
    mel = mel_tensor.clone().detach().cpu()
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-5)  # Normalize to [0, 1]
    mel = mel.unsqueeze(0)  # Add batch dimension
    writer.add_image(tag, mel, global_step=step)


if __name__ == "__main__":
    if t_setup_only:
        print("🔧 Setup mode: building manifest and indexing phoneme dictionary...")
        setup()
        print("✅ Setup complete.")
        sys.exit(0)
    if t_train_only:
        print("🚀 Training mode: starting training...")
        if not os.path.exists(manifest_path):
            print(
                f"❗ Manifest file not found at {manifest_path}. Please run setup first."
            )
            sys.exit(1)
    if not os.path.exists(dict_path):
        print(
            f"❗ Phoneme dictionary not found at {dict_path}. Please run setup first."
        )
        sys.exit(1)
    if not os.path.exists(samples_path):
        print(f"❗ Samples directory not found at {samples_path}")
        sys.exit(1)
    if not os.path.exists(ckpt_dir):
        print(f"❗ Checkpoint directory not found at {ckpt_dir}. Creating it.")
        os.makedirs(ckpt_dir)
