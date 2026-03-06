#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def stats(x: np.ndarray):
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    return peak, rms, int(x.size)


def load_mono_resample(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True)
    x = x.astype(np.float32)
    x = x.mean(axis=1)

    if sr != target_sr:
        t = torch.from_numpy(x)[None, :]
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=target_sr)
        x = t[0].cpu().numpy().astype(np.float32)
        sr = target_sr

    mx = float(np.max(np.abs(x))) if x.size else 0.0
    if mx > 1.5:
        x = x / (mx + 1e-12)

    return x, sr


def write_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr)


def try_model_call(m, x: torch.Tensor):
    """
    Try common TorchScript callable patterns for IIL RAVE exports.
    Returns audio tensor y of shape (T,) on CPU float32.
    """
    # x: (1,1,T)
    candidates = []

    # Most likely: m(x) -> (1,1,T) or tuple
    candidates.append(("m(x)", lambda: m(x)))

    # Sometimes forward has extra args (e.g., fidelity)
    # Try a couple of scalars
    for fid in (0.95, 0.99, 0.9):
        candidates.append((f"m(x, {fid})", lambda fid=fid: m(x, fid)))

    # If methods exist: encode/decode
    method_names = []
    try:
        method_names = list(m._c._method_names())
    except Exception:
        pass

    if "encode" in method_names and "decode" in method_names:
        candidates.append(("decode(encode(x))", lambda: m.decode(m.encode(x))))

    # Some exports use `analysis` / `synthesis` naming
    if "analysis" in method_names and "synthesis" in method_names:
        candidates.append(("synthesis(analysis(x))", lambda: m.synthesis(m.analysis(x))))

    last_err = None
    for label, fn in candidates:
        try:
            y = fn()
            return label, y
        except Exception as e:
            last_err = (label, e)
            continue

    # If everything failed, print useful info and raise
    info = []
    try:
        info.append(f"forward.schema: {m.forward.schema}")
    except Exception:
        pass
    try:
        info.append(f"methods: {m._c._method_names()}")
    except Exception:
        pass

    msg = "All TorchScript call patterns failed.\n"
    if last_err:
        msg += f"Last tried: {last_err[0]} -> {repr(last_err[1])}\n"
    if info:
        msg += "\n".join(info) + "\n"
    raise RuntimeError(msg)


def extract_audio(y) -> np.ndarray:
    """
    Accept tensor/tuple/list outputs and return mono waveform np.float32 (T,)
    """
    if isinstance(y, (tuple, list)):
        # take first tensor-like
        for item in y:
            if isinstance(item, torch.Tensor):
                y = item
                break

    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"Unexpected model output type: {type(y)}")

    y = y.detach().cpu()

    # expected shapes: (1,1,T) or (1,T) or (T,)
    if y.ndim == 3:
        y = y[0, 0, :]
    elif y.ndim == 2:
        y = y[0, :]
    elif y.ndim == 1:
        pass
    else:
        raise RuntimeError(f"Unexpected output shape: {tuple(y.shape)}")

    out = y.numpy().astype(np.float32)

    mx = float(np.max(np.abs(out))) if out.size else 0.0
    if mx > 1.5:
        out = out / (mx + 1e-12)

    return out


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_one(args, model_id: str, in_path: Path, out_path: Path, test_type: str):
    wav, sr = load_mono_resample(in_path, args.sr)
    in_peak, in_rms, n = stats(wav)
    print(f"\n=== INPUT ===\n{in_path}\nsr={sr} samples={n} peak={in_peak:.4f} rms={in_rms:.4f}")

    device = "cuda" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    m = torch.jit.load(str(args.model), map_location=device)
    m.eval()

    x = torch.from_numpy(wav)[None, None, :].to(device)

    with torch.inference_mode():
        label, y_raw = try_model_call(m, x)
        print(f"[MODEL CALL] {label}")


    out = extract_audio(y_raw)
    out_peak, out_rms, n2 = stats(out)
    print(f"\n=== OUTPUT ===\n{out_path}\nsr={sr} samples={n2} peak={out_peak:.4f} rms={out_rms:.4f}")

    write_wav(out_path, out, sr)

    if args.log_csv:
        append_csv(Path(args.log_csv), {
            "exp_id": args.exp_id,
            "model_id": model_id,
            "model_path": str(Path(args.model).resolve()),
            "test_type": test_type,
            "input_file": str(in_path.resolve()),
            "output_file": str(out_path.resolve()),
            "sr": sr,
            "in_peak": f"{in_peak:.6f}",
            "in_rms": f"{in_rms:.6f}",
            "out_peak": f"{out_peak:.6f}",
            "out_rms": f"{out_rms:.6f}",
            "notes": f"call={label}"
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="E02_pretrained_rave_nature_prior")
    ap.add_argument("--model", required=True, help="TorchScript model .ts")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--gpu", type=int, default=0, help=">=0 uses CUDA if available; -1 uses CPU")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch_in", default=None)
    ap.add_argument("--batch_out", default=None)
    ap.add_argument("--test_type", default="C_office_through_model")
    ap.add_argument("--log_csv", default="experiments/E02_pretrained_rave_nature_prior/logs/bench.csv")
    args = ap.parse_args()

    model_id = args.model_id or Path(args.model).stem

    if args.input and args.batch_in:
        raise SystemExit("Use either --input or --batch_in.")
    if args.batch_in and not args.batch_out:
        raise SystemExit("--batch_out required with --batch_in")

    if args.input:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else Path(f"experiments/E02_pretrained_rave_nature_prior/outputs/{in_path.stem}__{model_id}__{args.test_type}.wav")
        run_one(args, model_id, in_path, out_path, args.test_type)
        return

    if args.batch_in:
        in_dir = Path(args.batch_in)
        out_dir = Path(args.batch_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(in_dir.rglob("*.wav"))
        if not wavs:
            raise SystemExit(f"No wavs in {in_dir}")
        for p in wavs:
            rel = p.relative_to(in_dir)
            out_path = out_dir / rel.parent / f"{p.stem}__{model_id}__{args.test_type}.wav"
            run_one(args, model_id, p, out_path, args.test_type)
        return

    raise SystemExit("Provide --input or --batch_in")


if __name__ == "__main__":
    main()
