import os
import torchaudio
import numpy as np
import csv
from scipy.signal import welch

# === CONFIG ===
dataset_folder = "/home/shruti/rave-biophilic-sound-engine/data/raw/processed_classes_filtered/birds"
output_report = "/home/shruti/rave-biophilic-sound-engine/final_birds_quality_report.csv"

results = []

def compute_metrics(file_path):
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze().numpy()

    # RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # Noise floor (bottom 10%)
    sorted_abs = np.sort(np.abs(audio))
    noise_floor = np.mean(sorted_abs[:int(0.1 * len(sorted_abs))])

    # SNR (dB)
    snr = 20 * np.log10((rms + 1e-9) / (noise_floor + 1e-9))

    # Spectral features (Welch power spectral density)
    freqs, psd = welch(audio, sr, nperseg=2048)
    psd = psd + 1e-12

    # Spectral flatness (geometric mean / arithmetic mean)
    spectral_flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)

    # Spectral bandwidth (weighted frequency spread)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))

    return rms, noise_floor, snr, spectral_flatness, spectral_bandwidth

# Analyze all files
for file_name in sorted(os.listdir(dataset_folder)):
    if not file_name.endswith(".wav"):
        continue

    file_path = os.path.join(dataset_folder, file_name)
    rms, noise_floor, snr, flatness, bandwidth = compute_metrics(file_path)

    # Composite quality score (weighted)
    score = (snr * 0.5) + ((1 - flatness) * 20) + (bandwidth / 1000)

    results.append((file_name, rms, noise_floor, snr, flatness, bandwidth, score))

# Sort by quality score (highest = best)
results.sort(key=lambda x: x[6], reverse=True)

# Save full report
with open(output_report, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "rms", "noise_floor", "snr_dB", "spectral_flatness", "spectral_bandwidth", "quality_score"])
    writer.writerows(results)

print(f"Final quality report saved to: {output_report}")

print("\n=== 5 Best Clips (by quality score) ===")
for r in results[:5]:
    print(f"{r[0]} | SNR={r[3]:.2f} dB | Flatness={r[4]:.4f} | Bandwidth={r[5]:.1f} Hz")

print("\n=== 5 Worst Clips (by quality score) ===")
for r in results[-5:]:
    print(f"{r[0]} | SNR={r[3]:.2f} dB | Flatness={r[4]:.4f} | Bandwidth={r[5]:.1f} Hz")

