#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "test"
INPUT_DIR = DATA_DIR / "input_eval"
OUTPUT_DIR = DATA_DIR / "transformed_eval"

MODEL_DIR = EXP_ROOT / "models" / "iil_ts"
INFER_SCRIPT = SCRIPT_DIR / "rave_ts_infer.py"

LOG_DIR = EXP_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_CSV = LOG_DIR / "bird_batch_inference_eval_log.csv"

GPU_ID = "0"
TARGET_SR = "48000"
TEST_TYPE = "C_office_through_model"
MAX_RETRIES = 1

BIRD_MODELS = [
    "birds_dawnchorus_b2048_r48000_z8.ts",
    "birds_motherbird_b2048_r48000_z16.ts",
    "birds_pluma_b2048_r48000_z12.ts",
]

VALID_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def list_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def write_log_header_if_needed(csv_path: Path) -> None:
    if csv_path.exists():
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "input_file",
            "output_file",
            "attempt",
            "return_code",
            "output_exists",
            "status",
        ])


def append_log_row(
    csv_path: Path,
    model_name: str,
    input_file: Path,
    output_file: Path,
    attempt: int,
    result: subprocess.CompletedProcess,
    output_exists: bool,
    status: str,
) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            str(input_file),
            str(output_file),
            attempt,
            result.returncode,
            int(output_exists),
            status,
        ])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INFER_SCRIPT.exists():
        print(f"[ERROR] Missing inference script: {INFER_SCRIPT}")
        return 1

    input_files = list_input_files(INPUT_DIR)
    if not input_files:
        print(f"[ERROR] No input files found in: {INPUT_DIR}")
        return 1

    print(f"[INFO] Found {len(input_files)} eval input files")
    write_log_header_if_needed(RUN_LOG_CSV)

    failures = 0
    successes = 0

    for model_file in BIRD_MODELS:
        model_path = MODEL_DIR / model_file
        if not model_path.exists():
            print(f"[ERROR] Missing model: {model_path}")
            failures += len(input_files)
            continue

        model_name = model_file.replace(".ts", "")
        model_output_dir = OUTPUT_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print(f"[MODEL] {model_name}")
        print("==============================")

        for input_file in input_files:
            output_file = model_output_dir / f"{input_file.stem}__{model_name}.wav"

            if output_file.exists():
                print(f"[SKIP] {output_file.name}")
                successes += 1
                continue

            cmd = [
                sys.executable,
                str(INFER_SCRIPT),
                "--model", str(model_path),
                "--input", str(input_file),
                "--output", str(output_file),
                "--test_type", TEST_TYPE,
                "--sr", TARGET_SR,
                "--gpu", GPU_ID,
            ]

            ok = False

            for attempt in range(1, MAX_RETRIES + 2):
                print(f"[RUN] {input_file.name} -> {model_name} (attempt {attempt})")
                result = run_command(cmd)
                output_exists = output_file.exists()

                if result.returncode == 0 and output_exists:
                    append_log_row(
                        RUN_LOG_CSV,
                        model_name,
                        input_file,
                        output_file,
                        attempt,
                        result,
                        output_exists,
                        "OK",
                    )
                    ok = True
                    successes += 1
                    break

                append_log_row(
                    RUN_LOG_CSV,
                    model_name,
                    input_file,
                    output_file,
                    attempt,
                    result,
                    output_exists,
                    "RETRY" if attempt <= MAX_RETRIES else "FAILED",
                )

            if not ok:
                print(f"[FAIL] {input_file.name} -> {model_name}")
                failures += 1

    print("\n[DONE]")
    print(f"Successes: {successes}")
    print(f"Failures : {failures}")
    print(f"Log file : {RUN_LOG_CSV}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
