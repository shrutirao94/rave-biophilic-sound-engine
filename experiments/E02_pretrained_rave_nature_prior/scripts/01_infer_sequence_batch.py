#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys
import time

REPO_ROOT = Path("/home/shruti/rave-biophilic-sound-engine")
INPUT_DIR = REPO_ROOT / "data/test/input/sequence"
OUTPUT_ROOT = REPO_ROOT / "data/test/output/sequence"
MODEL_DIR = REPO_ROOT / "experiments/E02_pretrained_rave_nature_prior/models/iil_ts"
LOG_DIR = REPO_ROOT / "experiments/E02_pretrained_rave_nature_prior/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "birds_dawnchorus_b2048_r48000_z8": MODEL_DIR / "birds_dawnchorus_b2048_r48000_z8.ts",
    "water_pondbrain_b2048_r48000_z16": MODEL_DIR / "water_pondbrain_b2048_r48000_z16.ts",
}

# -------------------------------------------------------------------
# IMPORTANT:
# Replace build_command() with your ACTUAL inference command.
# For now this is a placeholder.
# -------------------------------------------------------------------
def build_command(model_path: Path, input_path: Path, output_path: Path) -> list[str]:
    raise NotImplementedError(
        "Please replace build_command() with your real TorchScript inference command."
    )


def run_one(model_name: str, model_path: Path, input_path: Path, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{model_name}__{input_path.stem}.log"

    try:
        cmd = build_command(model_path, input_path, output_path)
    except NotImplementedError as e:
        print(f"[CONFIG ERROR] {e}")
        return False

    print(f"\n[MODEL ] {model_name}")
    print(f"[INPUT ] {input_path}")
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
        return False


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob("*.wav"))
    if not input_files:
        raise FileNotFoundError(f"No .wav files found in: {INPUT_DIR}")

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

    print("=== Sequence Batch Inference ===")
    print(f"Input dir   : {INPUT_DIR}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Models      : {list(MODELS.keys())}")
    print(f"Num inputs  : {len(input_files)}")

    total = 0
    passed = 0
    failed = 0

    for model_name, model_path in MODELS.items():
        model_out_dir = OUTPUT_ROOT / model_name

        for input_path in input_files:
            total += 1
            output_path = model_out_dir / input_path.name

            if output_path.exists():
                print(f"\n[SKIP  ] already exists: {output_path}")
                passed += 1
                continue

            ok = run_one(model_name, model_path, input_path, output_path)
            if ok:
                passed += 1
            else:
                failed += 1

    print("\n=== SUMMARY ===")
    print(f"Total jobs : {total}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")


if __name__ == "__main__":
    main()
