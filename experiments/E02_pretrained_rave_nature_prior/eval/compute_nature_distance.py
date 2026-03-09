#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BENCH_CSV = EXP_ROOT / "bench_eval.csv"
REFERENCE_DIR = PROJECT_ROOT / "data" / "test" / "reference_eval" / "birds"
OUT_CSV = EXP_ROOT / "results" / "nature_distance_eval_results.csv"

SR = 48000
N_MFCC = 20
EPS = 1e-6


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def extract_feature_vector(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio)
    rms = librosa.feature.rms(y=audio)

    extra = np.array([
        np.mean(centroid),
        np.mean(bandwidth),
        np.mean(flatness),
        np.mean(rms),
    ], dtype=np.float32)

    return np.concatenate([mfcc_mean.astype(np.float32), extra], axis=0)


def list_reference_files(reference_dir: Path) -> list[Path]:
    return sorted(reference_dir.glob("*.wav"))


def compute_reference_distribution(reference_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    feats = []

    for path in reference_files:
        y = load_audio(path)
        feats.append(extract_feature_vector(y))

    X = np.stack(feats, axis=0)
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * EPS
    cov_inv = np.linalg.inv(cov)

    return mu, cov_inv


def main() -> int:
    if not BENCH_CSV.exists():
        print(f"[ERROR] Missing bench file: {BENCH_CSV}")
        return 1

    if not REFERENCE_DIR.exists():
        print(f"[ERROR] Missing reference dir: {REFERENCE_DIR}")
        return 1

    ref_files = list_reference_files(REFERENCE_DIR)
    if not ref_files:
        print(f"[ERROR] No reference wav files in: {REFERENCE_DIR}")
        return 1

    print(f"[INFO] Found {len(ref_files)} bird reference clips")
    print("[INFO] Computing bird reference distribution...")

    mu_ref, cov_inv_ref = compute_reference_distribution(ref_files)

    df = pd.read_csv(BENCH_CSV)
    results = []

    print(f"[INFO] Processing {len(df)} eval rows...")

    for i, row in df.iterrows():
        input_file = Path(row["input_file"])
        output_file = Path(row["output_file"])

        if not input_file.exists():
            print(f"[WARN] Missing input: {input_file}")
            continue

        if not output_file.exists():
            print(f"[WARN] Missing output: {output_file}")
            continue

        y_in = load_audio(input_file)
        y_out = load_audio(output_file)

        f_in = extract_feature_vector(y_in)
        f_out = extract_feature_vector(y_out)

        d_in = mahalanobis(f_in, mu_ref, cov_inv_ref)
        d_out = mahalanobis(f_out, mu_ref, cov_inv_ref)
        delta = d_out - d_in

        results.append({
            "model_id": row["model_id"],
            "input_file": str(input_file),
            "output_file": str(output_file),
            "reference_class": row["reference_class"],
            "distance_input_to_birds": float(d_in),
            "distance_output_to_birds": float(d_out),
            "delta_nature_distance": float(delta),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            print(f"[{i+1}/{len(df)}] processed")

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(out_df)} rows to {OUT_CSV}")

    if len(out_df):
        print("\nMean delta by model:")
        print(out_df.groupby("model_id")["delta_nature_distance"].agg(["mean", "std", "count"]).sort_values("mean"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
