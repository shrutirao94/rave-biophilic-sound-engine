import os
import random
import shutil

# Paths relative to project root
INPUT_DIR = "data/raw/processed_curated_filtered"
OUTPUT_DIR = "data/raw/processed_balanced_filtered"
MAX_FILES_PER_CLASS = 150

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Balance each subfolder
for category in os.listdir(INPUT_DIR):
    in_path = os.path.join(INPUT_DIR, category)
    out_path = os.path.join(OUTPUT_DIR, category)

    if not os.path.isdir(in_path):
        continue

    os.makedirs(out_path, exist_ok=True)

    wavs = [f for f in os.listdir(in_path) if f.endswith(".wav")]
    selected = random.sample(wavs, min(MAX_FILES_PER_CLASS, len(wavs)))

    for f in selected:
        shutil.copy(os.path.join(in_path, f), os.path.join(out_path, f))

    print(f"✅ {category}: {len(selected)} files copied")

