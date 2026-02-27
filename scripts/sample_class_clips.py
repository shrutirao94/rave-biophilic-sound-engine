import os
import random
import torchaudio
import matplotlib
matplotlib.use('Agg')  # Save plots instead of displaying them
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
data_dir = "data/raw/processed_classes_filtered/birds"
num_samples = 5
output_dir = "sampled_bird_clips"
os.makedirs(output_dir, exist_ok=True)

# Get all WAV files
all_files = [os.path.join(root, f) 
             for root, _, files in os.walk(data_dir) 
             for f in files if f.endswith(".wav")]

print(f"Found {len(all_files)} bird clips.")
if len(all_files) < num_samples:
    num_samples = len(all_files)

sample_files = random.sample(all_files, num_samples)
print(f"Sampling {num_samples} clips...")

for i, file_path in enumerate(sample_files):
    wave, sr = torchaudio.load(file_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)  # Convert to mono

    # Compute loudness (RMS in dB)
    rms = wave.pow(2).mean().sqrt().item()
    rms_db = 20 * np.log10(rms + 1e-9)

    # Plot spectrogram
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )(wave)

    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.squeeze(0).numpy(), aspect='auto', origin='lower')
    plt.title(f"Clip {i+1}: {os.path.basename(file_path)}\nLoudness: {rms_db:.2f} dB")
    plt.xlabel("Frames")
    plt.ylabel("Mel bins")
    plt.colorbar(label="dB")
    out_path = os.path.join(output_dir, f"sample_{i+1}.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[{i+1}/{num_samples}] Saved spectrogram: {out_path} (Loudness: {rms_db:.2f} dB)")

print(f"\n✅ Done. Spectrograms saved in: {output_dir}")

