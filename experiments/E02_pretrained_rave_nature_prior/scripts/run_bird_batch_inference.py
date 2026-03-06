#!/usr/bin/env python3

import subprocess
from pathlib import Path
import sys

# --------------------------------------------------
# Paths relative to this script
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent

DATA_DIR = EXP_ROOT / "data" / "test"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "transformed"

MODEL_DIR = EXP_ROOT / "models" / "iil_ts"

INFER_SCRIPT = SCRIPT_DIR / "rave_ts_infer.py"

GPU_ID = "0"
TARGET_SR = "48000"

TEST_TYPE = "C_office_through_model"

# --------------------------------------------------
# Bird models only
# --------------------------------------------------

BIRD_MODELS = [
    "birds_dawnchorus_b2048_r48000_z8.ts",
    "birds_motherbird_b2048_r48000_z16.ts",
    "birds_pluma_b2048_r48000_z12.ts",
]

# --------------------------------------------------

def get_input_files():

    exts = {".wav", ".mp3", ".flac", ".ogg"}

    return sorted(
        [
            p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in exts
        ]
    )


def run_cmd(cmd):

    print("\n[RUN]")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    return result.returncode


def main():

    input_files = get_input_files()

    if not input_files:
        print("No input files found.")
        return

    print(f"Found {len(input_files)} office clips")

    for model_file in BIRD_MODELS:

        model_path = MODEL_DIR / model_file

        model_name = model_file.replace(".ts", "")

        model_output_dir = OUTPUT_DIR / model_name

        model_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n================================")
        print("MODEL:", model_name)
        print("================================")

        for input_file in input_files:

            output_file = model_output_dir / f"{input_file.stem}__{model_name}.wav"

            cmd = [
                sys.executable,
                str(INFER_SCRIPT),
                "--model",
                str(model_path),
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--test_type",
                TEST_TYPE,
                "--sr",
                TARGET_SR,
                "--gpu",
                GPU_ID,
            ]

            code = run_cmd(cmd)

            if code != 0:

                print("FAILED:", input_file)

            else:

                print("OK:", output_file)


if __name__ == "__main__":
    main()
