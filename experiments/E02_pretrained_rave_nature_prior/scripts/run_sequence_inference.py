#!/usr/bin/env python3

import os
import subprocess
import sys
import time
from pathlib import Path

# Reduce thread-related instability in native libs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch


REPO_ROOT = Path("/home/shruti/rave-biophilic-sound-engine")

INFER_SCRIPT = REPO_ROOT / "experiments/E02_pretrained_rave_nature_prior/scripts/run_inference.py"
INPUT_DIR = REPO_ROOT / "data/test/input/sequence"
OUTPUT_ROOT = REPO_ROOT / "data/test/output/sequence"
MODEL_DIR = REPO_ROOT / "experiments/E02_pretrained_rave_nature_prior/models/iil_ts"
LOG_DIR = REPO_ROOT / "experiments/E02_pretrained_rave_nature_prior/logs/sequence_inference"

SR = 48000
GPU = "0"

MODELS = {
    "birds_dawnchorus_b2048_r48000_z8": MODEL_DIR / "birds_dawnchorus_b2048_r48000_z8.ts",
    "water_pondbrain_b2048_r48000_z16": MODEL_DIR / "water_pondbrain_b2048_r48000_z16.ts",
}


def build_command(model_path: Path, input_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(INFER_SCRIPT),
        "--model", str(model_path),
        "--input", str(input_path),
        "--output", str(output_path),
        "--sr", str(SR),
        "--gpu", GPU,
    ]


def run_job(model_name: str, model_path: Path, input_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{model_name}__{input_path.stem}.log"
    cmd = build_command(model_path, input_path, output_path)

    print("\n==================================================")
    print(f"[MODEL ] {model_name}")
    print(f"[INPUT ] {input_path.name}")
    print(f"[OUTPUT] {output_path}")
    print(f"[LOG   ] {log_path}")
    print(f"[CMD   ] {' '.join(cmd)}")

    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        elapsed = time.time() - t0

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("COMMAND:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout or "")
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr or "")
            f.write(f"\n\nELAPSED_SEC: {elapsed:.2f}\n")

        if not output_path.exists():
            print("[STATUS] FAIL")
            print("[ERROR ] Command finished but output file was not created.")
            return False

        print(f"[STATUS] PASS ({elapsed:.2f}s)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("COMMAND:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("STDOUT:\n")
            f.write(e.stdout or "")
            f.write("\n\nSTDERR:\n")
            f.write(e.stderr or "")
            f.write(f"\n\nELAPSED_SEC: {elapsed:.2f}\n")
            f.write(f"RETURN_CODE: {e.returncode}\n")

        print(f"[STATUS] FAIL ({elapsed:.2f}s)")
        print(f"[ERROR ] returncode={e.returncode}")

        # -11 is the usual Unix code for segmentation fault
        if e.returncode == -11:
            print("[ERROR ] Segmentation fault detected.")
        return False


def main():
    if not INFER_SCRIPT.exists():
        raise FileNotFoundError(f"Inference script not found: {INFER_SCRIPT}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob("*.wav"))
    if not input_files:
        raise FileNotFoundError(f"No wav files found in: {INPUT_DIR}")

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== SET 1: SEQUENCE INFERENCE ===")
    print(f"Input dir   : {INPUT_DIR}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Log dir     : {LOG_DIR}")
    print(f"Models      : {list(MODELS.keys())}")
    print(f"Num inputs  : {len(input_files)}")
    print(f"Sample rate : {SR}")
    print(f"GPU         : {GPU}")
    print("Execution   : sequential, one file at a time")

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for model_name, model_path in MODELS.items():
        model_out_dir = OUTPUT_ROOT / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n########## MODEL: {model_name} ##########")

        for input_path in input_files:
            total += 1
            output_path = model_out_dir / input_path.name

            if output_path.exists():
                print(f"\n[SKIP  ] already exists: {output_path}")
                skipped += 1
                continue

            ok = run_job(model_name, model_path, input_path, output_path)

            # Reduce accumulation of GPU memory pressure between runs
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Small pause between jobs to reduce stress on runtime / driver
            time.sleep(1.0)

            if ok:
                passed += 1
            else:
                failed += 1
                print("\n[STOP  ] Stopping batch after first failure to avoid repeated crashes.")
                print("Check the corresponding log file before continuing.")
                print(f"Log dir: {LOG_DIR}")
                print("\n=== SUMMARY (EARLY STOP) ===")
                print(f"Total jobs attempted : {total}")
                print(f"Passed               : {passed}")
                print(f"Failed               : {failed}")
                print(f"Skipped              : {skipped}")
                return

    print("\n=== SUMMARY ===")
    print(f"Total jobs : {total}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")
    print(f"Skipped    : {skipped}")


if __name__ == "__main__":
    main()
