import os
import subprocess
from pathlib import Path

# === CONFIG ===
input_dir = "data/raw/office"
model_path = "trained_models/synth_wind/rave_synth_wind_streaming.ts"
output_dir = "new_sounds/synth_wind/office/"
batch_size = 5
gpu = 0  # use -1 for CPU, 0 for GPU
chunk_size = 150000  # as per your command

# Ensure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Get list of .wav files
files = sorted(Path(input_dir).rglob("*.wav"))

for i in range(0, len(files), batch_size):
    batch = files[i:i + batch_size]
    print(f"\n=== Processing batch {i // batch_size + 1} ({len(batch)} files) ===")
    # Build command
    cmd = ["rave", "generate","--model", model_path,"--out_path", output_dir,"--gpu", str(gpu),"--stream"]
    # Add each file as input
    for f in batch:
        cmd.extend(["--input", str(f)])
        # Run the command
        subprocess.run(cmd)

