#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REFERENCE_RAW_DIR = PROJECT_ROOT / "data" / "test" / "reference"
REFERENCE_EVAL_DIR = PROJECT_ROOT / "data" / "test" / "reference_eval"

TARGET_SR = 48000
SEGMENT_SEC = 10.0
MIN_RMS = 0.003  # skip near-silent segments

VALID_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def canonical_class_name(name: str) -> str:
    if name == "bird":
        return "birds"
    return name


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def load_audio(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)

    if y.ndim == 2:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y.astype(np.float32), target_sr


def segment_audio(y: np.ndarray, sr: int, segment_sec: float) -> list[np.ndarray]:
    seg_len = int(round(segment_sec * sr))
    n_segments = len(y) // seg_len

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        seg = y[start:end]
        if len(seg) == seg_len:
            segments.append(seg)
    return segments


def list_audio_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def process_class_folder(class_dir: Path) -> None:
    raw_class_name = class_dir.name
    out_class_name = canonical_class_name(raw_class_name)
    out_dir = REFERENCE_EVAL_DIR / out_class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(class_dir)
    if not files:
        print(f"[WARN] No audio files in {class_dir}")
        return

    print("\n========================================")
    print(f"[CLASS] {raw_class_name} -> {out_class_name}")
    print("========================================")

    total_written = 0
    total_skipped = 0

    for path in files:
        print(f"[PROCESS] {path.name}")

        try:
            y, sr = load_audio(path, TARGET_SR)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue

        segments = segment_audio(y, sr, SEGMENT_SEC)
        print(f"  loaded: {len(y)} samples @ {sr} Hz")
        print(f"  segments: {len(segments)} x {SEGMENT_SEC:.1f}s")

        written_this_file = 0

        for idx, seg in enumerate(segments):
            seg_rms = rms(seg)

            if seg_rms < MIN_RMS:
                total_skipped += 1
                continue

            out_name = f"{path.stem}_seg{idx:03d}.wav"
            out_path = out_dir / out_name
            sf.write(str(out_path), seg, sr)

            total_written += 1
            written_this_file += 1

        print(f"  wrote: {written_this_file}")

    print(f"\n[CLASS DONE] {out_class_name}")
    print(f"  written: {total_written}")
    print(f"  skipped: {total_skipped}")
    print(f"  outdir : {out_dir}")


def main() -> int:
    if not REFERENCE_RAW_DIR.exists():
        print(f"[ERROR] Missing raw reference dir: {REFERENCE_RAW_DIR}")
        return 1
    class_dirs = sorted([p for p in REFERENCE_RAW_DIR.iterdir() if p.is_dir()])

    if not class_dirs:
        print(f"[ERROR] No class folders found in: {REFERENCE_RAW_DIR}")
        return 1

    for class_dir in class_dirs:
        process_class_folder(class_dir)

    print("\n[DONE] Reference preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
