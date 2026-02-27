import torch
import torchaudio
import soundfile as sf
import os

# ==== Paths ====
MODEL_PATH = "../trained_modes/full_bal/full_bal.ts"
INPUT_PATH = "../data/test_audio/office/soft.wav"
OUTPUT_PATH = "../new_sounds/full_bal/transformed_soft.wav"
TARGET_SR = 44000  # ✅ Set explicitly to match training

# ==== Load model ====
print(f"🔄 Loading model from: {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH)
model.eval()

# ==== Load input audio ====
wave, sr = torchaudio.load(INPUT_PATH)
print(f"🔊 Loaded: {INPUT_PATH}, Sample Rate: {sr}, Shape: {wave.shape}")

# ==== Resample if needed ====
if sr != TARGET_SR:
    print(f"🔁 Resampling from {sr} Hz → {TARGET_SR} Hz")
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
    wave = resampler(wave)
    sr = TARGET_SR

# ==== Ensure mono ====
if wave.shape[0] > 1:
    wave = wave.mean(dim=0, keepdim=True)

# ==== Chunking & Transform ====
segment_len = 2**14
num_segments = wave.shape[-1] // segment_len
print(f"📦 Total segments: {num_segments}")
output = []

with torch.no_grad():
    for i in range(num_segments):
        segment = wave[:, i*segment_len : (i+1)*segment_len]
        segment = segment.unsqueeze(0)  # Shape: [1, 1, 16384]
        z = model.encode(segment)
        recon = model.decode(z)
        output.append(recon.squeeze(0))

# ==== Concatenate and Save ====
output_waveform = torch.cat(output, dim=-1).cpu().numpy()
sf.write(OUTPUT_PATH, output_waveform.T, TARGET_SR)  # Transpose if needed
print(f"✅ Transformed audio saved to: {OUTPUT_PATH}")

