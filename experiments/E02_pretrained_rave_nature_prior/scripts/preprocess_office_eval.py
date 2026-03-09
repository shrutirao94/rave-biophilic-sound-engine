#!/usr/bin/env python3

from pathlib import Path
import librosa
import soundfile as sf

TARGET_SR = 48000
SEG_SECONDS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data/test/input"
OUTPUT_DIR = PROJECT_ROOT / "data/test/input_eval"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_file(path):

    print(f"Processing {path.name}")

    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    seg_len = TARGET_SR * SEG_SECONDS
    n_segments = len(y) // seg_len

    for i in range(n_segments):

        start = i * seg_len
        end = start + seg_len

        segment = y[start:end]

        outname = f"{path.stem}_seg{i:03d}.wav"
        outpath = OUTPUT_DIR / outname

        sf.write(outpath, segment, TARGET_SR)

def main():

    files = sorted(INPUT_DIR.glob("*.wav"))

    print("Input files:", len(files))

    for f in files:
        process_file(f)

    total = len(list(OUTPUT_DIR.glob("*.wav")))
    print("Segments created:", total)

if __name__ == "__main__":
    main()
