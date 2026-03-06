#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import math

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "test"

# Raw/original data
INPUT_RAW_DIR = DATA_ROOT / "input"
REFERENCE_BIRD_RAW_DIR = DATA_ROOT / "reference" / "bird"

# Standardised evaluation data
INPUT_EVAL_DIR = DATA_ROOT / "input_eval"
REFERENCE_BIRD_EVAL_DIR = DATA_ROOT / "reference_eval" / "birds"

TARGET_SR = 48000
SEGMENT_SEC = 10.0

# Bird reference filtering
MIN_RMS_FOR_REFERENCE = 0.005  # skip very quiet/silent bird segments

VALID_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def list_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def load_audio_mono_48k(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y = y.astype(np.float32)
    return y, target_sr


def segment_audio(y: np.ndarray, sr: int, segment_sec: float) -> list[np.ndarray]:
    seg_len = int(round(segment_sec * sr))
    if seg_len <= 0:
        raise ValueError("segment_sec must be > 0")

    total = len(y)
    n_segments = total // seg_len
    segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        seg = y[start:end]
        if len(seg) == seg_len:
            segments.append(seg)

    return segments


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def write_segments(
    input_files: list[Path],
    out_dir: Path,
    segment_sec: float,
    filter_quiet: bool = False,
    min_rms: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in input_files:
        print(f"\n[PROCESS] {path}")
        y, sr = load_audio_mono_48k(path, TARGET_SR)
        segments = segment_audio(y, sr, segment_sec)

        print(f"  loaded: {len(y)} samples @ {sr} Hz")
        print(f"  segments: {len(segments)} x {segment_sec:.1f}s")

        kept = 0

        for idx, seg in enumerate(segments):
            seg_rms = rms(seg)

            if filter_quiet and seg_rms < min_rms:
                print(f"    skip seg {idx:03d} | rms={seg_rms:.6f}")
                continue

            out_name = f"{path.stem}_seg{idx:03d}.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), seg, sr)
            kept += 1

            print(f"    wrote {out_name} | rms={seg_rms:.6f}")

        print(f"  kept: {kept}/{len(segments)}")


def main() -> int:
    office_files = list_audio_files(INPUT_RAW_DIR)
    bird_files = list_audio_files(REFERENCE_BIRD_RAW_DIR)

    if not office_files:
        print(f"[ERROR] No office files found in {INPUT_RAW_DIR}")
        return 1

    if not bird_files:
        print(f"[ERROR] No bird reference files found in {REFERENCE_BIRD_RAW_DIR}")
        return 1

    print("\n=== PREPARING OFFICE EVAL CLIPS ===")
    write_segments(
        input_files=office_files,
        out_dir=INPUT_EVAL_DIR,
        segment_sec=SEGMENT_SEC,
        filter_quiet=False,
        min_rms=0.0,
    )

    print("\n=== PREPARING BIRD REFERENCE CLIPS ===")
    write_segments(
        input_files=bird_files,
        out_dir=REFERENCE_BIRD_EVAL_DIR,
        segment_sec=SEGMENT_SEC,
        filter_quiet=True,
        min_rms=MIN_RMS_FOR_REFERENCE,
    )

    print("\nDone.")
    print(f"Office eval clips: {INPUT_EVAL_DIR}")
    print(f"Bird reference eval clips: {REFERENCE_BIRD_EVAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
