import torch
import torchaudio
import gin
from rave.model import RAVE

# === Config ===
gin.parse_config_file("archive/nature_rave_be81ca5504/config.gin")  # adjust if needed
ckpt_path = "models/nature_rave/nature_rave_50241f6005/version_1/checkpoints/best.ckpt"
test_wav_path = "data/raw/processed_balanced_filtered/birds/st_augustin_woodpecker_forest_seg0006.wav"
out_path = "new_sounds/full_bal/reconstructed_ckpt.wav"

# === Load Model ===
print("Loading model from checkpoint...")
model = RAVE.load_from_checkpoint(ckpt_path)
model.eval()

# === Load Audio ===
print("Loading audio...")
wave, sr = torchaudio.load(test_wav_path)
print(f"Sample rate: {sr}, shape: {wave.shape}")

# Ensure mono
if wave.shape[0] > 1:
    print("Converting to mono...")
    wave = wave.mean(dim=0, keepdim=True)

# Add batch dimension
wave = wave.unsqueeze(0)  # [1, 1, T]

# === Encode & Decode ===
print("Encoding and decoding...")
with torch.no_grad():
    z = model.encode(wave)
    z = z[:, :128, :]
    recon = model.decode(z)

# === Save ===
print(f"Saving to: {out_path}")
torchaudio.save(out_path, recon.squeeze(0), sr)
print("Done.")

