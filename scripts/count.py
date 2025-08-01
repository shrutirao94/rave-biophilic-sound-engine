import os

INPUT_DIR = "processed_curated_filtered/"

print(f"\n🎵 File counts per category in '{INPUT_DIR}':\n")
for category in os.listdir(INPUT_DIR):
    cat_path = os.path.join(INPUT_DIR, category)
    if os.path.isdir(cat_path):
        wav_files = [f for f in os.listdir(cat_path) if f.endswith(".wav")]
        print(f"📁 {category}: {len(wav_files)} files")

