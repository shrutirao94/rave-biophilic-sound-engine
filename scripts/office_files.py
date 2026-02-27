import os
import torchaudio
import torch
from pathlib import Path

# === CONFIG ===
input_file = "data/raw/office.mp3"
output_dir = "data/raw/office"
target_sr = 48000
segment_duration = 60.0  # seconds
segment_samples = int(target_sr * segment_duration)

os.makedirs(output_dir, exist_ok=True)

print(f"Processing long office file in chunks: {input_file}")

# Use torchaudio's streaming backend
torchaudio.set_audio_backend("sox_io")  # safer for large files
info = torchaudio.info(input_file)
orig_sr = info.sample_rate
num_frames = info.num_frames
channels = info.num_channels

print(f"Original SR: {orig_sr} Hz, Channels: {channels}, Total frames: {num_frames}")

# Calculate total segments
total_segments = (num_frames // (orig_sr * int(segment_duration))) + 1
print(f"Estimated segments: {total_segments}")

resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)

segment_idx = 1
frame_offset = 0
frames_per_chunk = orig_sr * int(segment_duration)  # read 60s chunks

while frame_offset < num_frames:
    out_path = Path(output_dir) / f"office_seg{segment_idx:03d}.wav"

    # === RESUME CHECK ===
    if out_path.exists():
        print(f"Skipping segment {segment_idx} (already exists)")
        frame_offset += frames_per_chunk
        segment_idx += 1
        continue

    # Read chunk
    wave, sr = torchaudio.load(input_file, frame_offset=frame_offset, num_frames=frames_per_chunk)
    frame_offset += wave.shape[1]

    # Convert to mono
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        wave = resampler(wave)

    # Pad if last chunk is short
    if wave.shape[1] < segment_samples:
        wave = torch.nn.functional.pad(wave, (0, segment_samples - wave.shape[1]))

    # Save file
    torchaudio.save(out_path, wave, target_sr)
    print(f"Saved {out_path}")

    segment_idx += 1

print(f"\n=== DONE ===\nSaved or skipped {segment_idx-1} segments in {output_dir}")

