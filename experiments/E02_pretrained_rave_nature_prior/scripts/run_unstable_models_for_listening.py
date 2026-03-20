#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data" / "test" / "input_eval"
OUTPUT_DIR = PROJECT_ROOT / "data" / "test" / "transformed_eval_unstable"
LOG_DIR = EXP_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

INFER_SCRIPT = SCRIPT_DIR / "rave_ts_infer.py"
MODEL_DIR = EXP_ROOT / "models" / "iil_ts"

LOG_CSV = LOG_DIR / "unstable_model_inference_log.csv"

TARGET_SR = "48000"
GPU_ID = "0"
TEST_TYPE = "C_office_through_model"
TIMEOUT_SEC = 180

MODELS = [
    "marinemammals_pondbrain_b2048_r48000_z20.ts",
    "humpbacks_pondbrain_b2048_r48000_z20.ts",
]

VALID_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def list_inputs(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def init_log(csv_path: Path) -> None:
    if csv_path.exists():
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_id",
            "input_file",
            "output_file",
            "return_code",
            "status",
            "stdout_tail",
            "stderr_tail",
        ])


def append_log(
    csv_path: Path,
    model_id: str,
    input_file: Path,
    output_file: Path,
    return_code: int | str,
    status: str,
    stdout: str,
    stderr: str,
) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_id,
            str(input_file),
            str(output_file),
            return_code,
            status,
            stdout[-2000:] if stdout else "",
            stderr[-2000:] if stderr else "",
        ])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    init_log(LOG_CSV)

    inputs = list_inputs(INPUT_DIR)
    if not inputs:
        print(f"[ERROR] No inputs found in {INPUT_DIR}")
        return 1

    print(f"[INFO] Found {len(inputs)} input clips")
    print(f"[INFO] Outputs will go to {OUTPUT_DIR}")

    total_ok = 0
    total_fail = 0

    for model_file in MODELS:
        model_path = MODEL_DIR / model_file
        if not model_path.exists():
            print(f"[ERROR] Missing model: {model_path}")
            continue

        model_id = model_file.replace(".ts", "")
        model_out_dir = OUTPUT_DIR / model_id
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print("\n========================================")
        print(f"[MODEL] {model_id}")
        print("========================================")

        ok = 0
        fail = 0

        for i, input_file in enumerate(inputs, start=1):
            output_file = model_out_dir / f"{input_file.stem}__{model_id}.wav"

            if output_file.exists():
                print(f"[SKIP] {i}/{len(inputs)} {output_file.name}")
                ok += 1
                total_ok += 1
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

            print(f"[RUN] {i}/{len(inputs)} {input_file.name}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SEC,
                )

                output_exists = output_file.exists()

                if result.returncode == 0 and output_exists:
                    print(f"[OK]   {output_file.name}")
                    append_log(
                        LOG_CSV,
                        model_id,
                        input_file,
                        output_file,
                        result.returncode,
                        "OK",
                        result.stdout,
                        result.stderr,
                    )
                    ok += 1
                    total_ok += 1
                else:
                    print(f"[FAIL] {input_file.name} rc={result.returncode}")
                    append_log(
                        LOG_CSV,
                        model_id,
                        input_file,
                        output_file,
                        result.returncode,
                        "FAILED",
                        result.stdout,
                        result.stderr,
                    )
                    fail += 1
                    total_fail += 1

            except subprocess.TimeoutExpired as e:
                print(f"[TIMEOUT] {input_file.name}")
                append_log(
                    LOG_CSV,
                    model_id,
                    input_file,
                    output_file,
                    "TIMEOUT",
                    "TIMEOUT",
                    e.stdout or "",
                    e.stderr or "",
                )
                fail += 1
                total_fail += 1

        print(f"\n[MODEL DONE] {model_id}")
        print(f"  succeeded: {ok}")
        print(f"  failed   : {fail}")

    print("\n========================================")
    print("[ALL DONE]")
    print("========================================")
    print(f"Total succeeded: {total_ok}")
    print(f"Total failed   : {total_fail}")
    print(f"Log file       : {LOG_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
