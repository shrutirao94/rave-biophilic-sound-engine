# scripts/check_segment_lengths.py
from pathlib import Path
import soundfile as sf

DATA_DIR = Path("data/preprocess/nature")  # Update if your folder is different
EXPECTED_DURATION = 5.0  # seconds
TOLERANCE = 0.05  # small tolerance for rounding or encoding

mismatches = []

for f in DATA_DIR.glob("*.wav"):
    with sf.SoundFile(f) as s:
        duration = len(s) / s.samplerate
        if abs(duration - EXPECTED_DURATION) > TOLERANCE:
            mismatches.append((f.name, duration))

if mismatches:
    print(f"⚠️ Found {len(mismatches)} mismatched segments:")
    for name, dur in mismatches:
        print(f"{name}: {dur:.3f}s")
else:
    print("✅ All segments are close to expected length.")

