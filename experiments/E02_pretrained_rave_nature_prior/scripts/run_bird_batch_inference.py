#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "test"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "transformed"
LOG_DIR = EXP_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = EXP_ROOT / "models" / "iil_ts"
INFER_SCRIPT = SCRIPT_DIR / "rave_ts_infer.py"

RUN_LOG_CSV = LOG_DIR / "bird_batch_inference_log.csv"

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


# --------------------------------------------------
# Helpers
# --------------------------------------------------

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
            "model_path",
            "input_file",
            "output_file",
            "attempt",
            "return_code",
            "output_exists",
            "stdout_tail",
            "stderr_tail",
            "status",
        ])


def append_log_row(
    csv_path: Path,
    model_name: str,
    model_path: Path,
    input_file: Path,
    output_file: Path,
    attempt: int,
    result: subprocess.CompletedProcess,
    output_exists: bool,
    status: str,
) -> None:
    stdout_tail = (result.stdout or "")[-2000:]
    stderr_tail = (result.stderr or "")[-2000:]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            str(model_path),
            str(input_file),
            str(output_file),
            attempt,
            result.returncode,
            int(output_exists),
            stdout_tail,
            stderr_tail,
            status,
        ])


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> int:
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("EXP_ROOT:", EXP_ROOT)
    print("INPUT_DIR:", INPUT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("INFER_SCRIPT:", INFER_SCRIPT)
    print("RUN_LOG_CSV:", RUN_LOG_CSV)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return 1

    if not MODEL_DIR.exists():
        print(f"[ERROR] Model directory not found: {MODEL_DIR}")
        return 1

    if not INFER_SCRIPT.exists():
        print(f"[ERROR] Inference script not found: {INFER_SCRIPT}")
        return 1

    input_files = list_input_files(INPUT_DIR)

    if not input_files:
        print(f"[ERROR] No input files found in {INPUT_DIR}")
        return 1

    print(f"[INFO] Found {len(input_files)} input files.")

    write_log_header_if_needed(RUN_LOG_CSV)

    successes = 0
    failures = 0

    for model_file in BIRD_MODELS:
        model_path = MODEL_DIR / model_file

        if not model_path.exists():
            print(f"[ERROR] Model missing: {model_path}")
            failures += len(input_files)
            continue

        model_name = model_file.replace(".ts", "")
        model_output_dir = OUTPUT_DIR / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n==================================================")
        print(f"[MODEL] {model_name}")
        print("==================================================")

        for input_file in input_files:
            output_file = model_output_dir / f"{input_file.stem}__{model_name}.wav"

            # Skip if already exists
            if output_file.exists():
                print(f"[SKIP] Already exists: {output_file}")
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

            final_status = "FAILED"

            for attempt in range(1, MAX_RETRIES + 2):
                print("\n[RUN]")
                print(" ".join(cmd))
                print(f"[ATTEMPT] {attempt}")

                result = run_command(cmd)

                if result.stdout:
                    print("\n[STDOUT]")
                    print(result.stdout)

                if result.stderr:
                    print("\n[STDERR]")
                    print(result.stderr)

                print(f"[RETURN CODE] {result.returncode}")

                output_exists = output_file.exists()

                # Treat as success only if return code is 0 and output exists
                if result.returncode == 0 and output_exists:
                    final_status = "OK"
                    append_log_row(
                        RUN_LOG_CSV,
                        model_name,
                        model_path,
                        input_file,
                        output_file,
                        attempt,
                        result,
                        output_exists,
                        final_status,
                    )
                    print(f"[OK] {output_file}")
                    successes += 1
                    break

                append_log_row(
                    RUN_LOG_CSV,
                    model_name,
                    model_path,
                    input_file,
                    output_file,
                    attempt,
                    result,
                    output_exists,
                    "RETRY" if attempt <= MAX_RETRIES else "FAILED",
                )

                if attempt <= MAX_RETRIES:
                    print("[WARN] Run failed, retrying once...")
                else:
                    print(f"[FAIL] {input_file.name} -> {model_name}")
                    failures += 1

            # end attempt loop

    print("\n==================================================")
    print("[DONE] Bird batch inference complete")
    print("==================================================")
    print(f"Successes: {successes}")
    print(f"Failures : {failures}")
    print(f"Log file : {RUN_LOG_CSV}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
