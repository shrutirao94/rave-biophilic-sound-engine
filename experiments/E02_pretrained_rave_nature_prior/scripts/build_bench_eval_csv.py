#!/usr/bin/env python3

from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "test"

INPUT_DIR = DATA_ROOT / "input_eval"
TRANSFORMED_DIR = DATA_ROOT / "transformed_eval"

OUT_CSV = PROJECT_ROOT / "experiments" / "E02_pretrained_rave_nature_prior" / "bench_eval.csv"


def main() -> None:
    rows = []

    for model_dir in sorted(TRANSFORMED_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        model_id = model_dir.name

        for wav in sorted(model_dir.glob("*.wav")):
            input_name = wav.name.split("__")[0] + ".wav"
            input_file = INPUT_DIR / input_name

            if not input_file.exists():
                print(f"[WARN] Missing input for output: {wav}")
                continue

            rows.append({
                "model_id": model_id,
                "input_file": str(input_file),
                "output_file": str(wav),
                "reference_class": "birds",
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_id", "input_file", "output_file", "reference_class"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
