#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import librosa
import soundfile as sf
import numpy as np


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--gpu", default="0")

    parser.add_argument("--test_type", default="inference")

    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    sr = args.sr

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    if not input_path.exists():
        raise RuntimeError(f"Input not found: {input_path}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n=== INPUT ===")
    print(input_path)

    y = load_audio(input_path, sr)

    print(f"sr={sr} samples={len(y)} peak={np.max(np.abs(y)):.4f} rms={np.sqrt(np.mean(y**2)):.4f}")

    x = torch.from_numpy(y)[None, None, :].to(device)

    model = torch.jit.load(str(model_path)).to(device)
    model.eval()

    with torch.no_grad():

        print("[MODEL CALL] m(x)")

        y_hat = model(x)

    y_hat = y_hat.squeeze().cpu().numpy()

    peak = np.max(np.abs(y_hat))
    rms = np.sqrt(np.mean(y_hat**2))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, y_hat, sr)

    print("\n=== OUTPUT ===")
    print(output_path)
    print(f"sr={sr} samples={len(y_hat)} peak={peak:.4f} rms={rms:.4f}")


if __name__ == "__main__":
    main()
