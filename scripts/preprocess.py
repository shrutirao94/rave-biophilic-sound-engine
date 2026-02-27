# preprocess.py
import subprocess
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/raw/nature")
audio_files = list(INPUT_DIR.rglob("*.mp3")) + list(INPUT_DIR.rglob("*.wav")) + list(INPUT_DIR.rglob("*.WAV"))


print(f"🎧 Found {len(audio_files)} files.")

for file_path in tqdm(audio_files):
    print(f"🔄 Processing: {file_path}")
    try:
        result = subprocess.run(
            ["python", "scripts/worker_preprocess.py", str(file_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"[SKIP] {file_path.name} failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"[FATAL] Could not run subprocess for {file_path.name}: {e}")

print("\n✅ Batch preprocessing complete.")

