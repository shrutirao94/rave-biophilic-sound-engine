# worker_preprocess.py
import sys
from pathlib import Path
import librosa
import soundfile as sf

SAMPLE_RATE = 48000
SEGMENT_DURATION = 60
MIN_SEGMENT_LENGTH = 1.0
OUTPUT_DIR = Path("data/preprocess/nature")

file_path = Path(sys.argv[1])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y = y / max(abs(y)) if max(abs(y)) > 0 else y

    total_samples = len(y)
    segment_samples = int(SEGMENT_DURATION * SAMPLE_RATE)

    for i in range(0, total_samples, segment_samples):
        segment = y[i:i+segment_samples]
        duration = len(segment) / SAMPLE_RATE

        if duration < MIN_SEGMENT_LENGTH:
            continue

        out_name = f"{file_path.stem}_seg{i // segment_samples:04d}.wav"
        out_path = OUTPUT_DIR / out_name
        sf.write(out_path, segment, SAMPLE_RATE)

except Exception as e:
    print(f"[ERROR] in worker for {file_path.name}: {e}")
    sys.exit(1)

