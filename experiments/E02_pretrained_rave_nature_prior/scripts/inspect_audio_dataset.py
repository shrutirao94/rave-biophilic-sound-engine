#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import csv
import sys

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "test"

INPUT_DIR = DATA_ROOT / "input"
REFERENCE_DIR = DATA_ROOT / "reference"
OUT_CSV = DATA_ROOT / "audio_inventory.csv"


def inspect_file(path: Path) -> dict:
    info = sf.info(str(path))
    duration_sec = info.frames / info.samplerate if info.samplerate else 0.0

    return {
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "samplerate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_sec": round(duration_sec, 3),
        "format": info.format,
        "subtype": info.subtype,
    }


def collect_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    files = collect_audio_files(INPUT_DIR) + collect_audio_files(REFERENCE_DIR)

    if not files:
        print(f"No audio files found under {DATA_ROOT}")
        return 1

    rows = []
    print("\n=== AUDIO INVENTORY ===\n")

    for path in files:
        try:
            row = inspect_file(path)
            rows.append(row)
            print(
                f"{row['name']}\n"
                f"  path: {row['path']}\n"
                f"  sr: {row['samplerate']} | ch: {row['channels']} | "
                f"frames: {row['frames']} | dur: {row['duration_sec']} s | "
                f"{row['format']} / {row['subtype']}\n"
            )
        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "name",
                "suffix",
                "samplerate",
                "channels",
                "frames",
                "duration_sec",
                "format",
                "subtype",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved inventory to: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
