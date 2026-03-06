#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout.strip())
    if p.stderr.strip():
        print("[stderr]")
        print(p.stderr.strip())
    return p.returncode

def main():
    print("=== E02 PREFLIGHT ===")
    print("python:", sys.version.replace("\n"," "))
    print("executable:", sys.executable)
    print("cwd:", os.getcwd())
    print("platform:", platform.platform())

    # GPU / torch
    try:
        import torch
        print("\n[torch]")
        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        print("cuda_version:", torch.version.cuda)
        print("device_count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("gpu0:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch import failed:", repr(e))

    # Audio deps
    try:
        import soundfile as sf
        print("\n[soundfile]", sf.__version__)
    except Exception as e:
        print("soundfile import failed:", repr(e))

    try:
        import torchaudio
        print("[torchaudio]", torchaudio.__version__)
    except Exception as e:
        print("torchaudio import failed:", repr(e))

    # RAVE import path
    try:
        import rave
        print("\n[rave]")
        print("rave_file:", rave.__file__)
        print("rave_dir:", str(Path(rave.__file__).parent))
    except Exception as e:
        print("rave import failed:", repr(e))

    # CLI identity
    run(["which", "rave"])
    run(["rave", "--help"])
    run(["rave", "export", "--help"])
    run(["rave", "generate", "--help"])

    # External tools
    run(["ffmpeg", "-version"])
    run(["sox", "--version"])
    run(["nvidia-smi"])

if __name__ == "__main__":
    main()
