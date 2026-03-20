#!/usr/bin/env python3
from pathlib import Path
import soundfile as sf

INPUT_DIR = Path("/home/shruti/rave-biophilic-sound-engine/data/test/input/sequence")
EXPECTED_SR = 48000
EXPECTED_EXT = ".wav"
EXPECTED_MIN_DURATION = 55   # seconds
EXPECTED_MAX_DURATION = 65


def check_wav(path: Path) -> tuple[bool, str]:
    try:
        info = sf.info(str(path))
    except Exception as e:
        return False, f"CORRUPTED/UNREADABLE: {e}"

    issues = []

    if info.samplerate != EXPECTED_SR:
        issues.append(f"sample_rate={info.samplerate} (expected {EXPECTED_SR})")

    if info.frames <= 0:
        issues.append("empty file (0 frames)")

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as e:
        return False, f"READ ERROR: {e}"

    if sr != EXPECTED_SR:
        issues.append(f"read_sr={sr} (expected {EXPECTED_SR})")

    if getattr(data, "size", 0) == 0:
        issues.append("empty audio array")

    if issues:
        return False, "; ".join(issues)

    duration = info.frames / info.samplerate
    if not (EXPECTED_MIN_DURATION <= duration <= EXPECTED_MAX_DURATION):
        issues.append(f"unexpected duration={duration:.2f}s")

    channels = info.channels
    subtype = info.subtype
    return True, f"OK | sr={info.samplerate} | ch={channels} | dur={duration:.2f}s | subtype={subtype}"


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"ERROR: input dir does not exist: {INPUT_DIR}")
        return

    wav_files = sorted(INPUT_DIR.glob(f"*{EXPECTED_EXT}"))

    if not wav_files:
        print(f"ERROR: no {EXPECTED_EXT} files found in {INPUT_DIR}")
        return

    print(f"Checking {len(wav_files)} files in: {INPUT_DIR}\n")

    ok_count = 0
    bad_count = 0

    for wav_path in wav_files:
        ok, msg = check_wav(wav_path)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {wav_path.name} -> {msg}")
        if ok:
            ok_count += 1
        else:
            bad_count += 1

    print("\n--- SUMMARY ---")
    print(f"PASS: {ok_count}")
    print(f"FAIL: {bad_count}")
    print(f"TOTAL: {len(wav_files)}")


if __name__ == "__main__":
    main()
