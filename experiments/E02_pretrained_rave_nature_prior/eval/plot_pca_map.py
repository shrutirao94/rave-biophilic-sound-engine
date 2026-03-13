#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BENCH_CSV = EXP_ROOT / "bench_eval.csv"
REFERENCE_DIR = PROJECT_ROOT / "data" / "test" / "reference_eval" / "birds"

FIG_DIR = EXP_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

OUT_PNG = FIG_DIR / "pca_acoustic_map.png"
OUT_CSV = FIG_DIR / "pca_acoustic_map_points.csv"

SR = 48000
EPS = 1e-6


# ----------------------------------------------------
# Audio loading
# ----------------------------------------------------

def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


# ----------------------------------------------------
# Feature extraction
# Use same feature family as compute_all_metrics.py
# ----------------------------------------------------

def extract_features(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)

    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
    flatness = librosa.feature.spectral_flatness(y=y)
    rms = librosa.feature.rms(y=y)
    onset = librosa.onset.onset_strength(y=y, sr=SR)

    feats = np.vstack([
        mfcc,
        centroid,
        bandwidth,
        flatness,
        rms,
        onset.reshape(1, -1),
    ])

    mean = np.mean(feats, axis=1)
    std = np.std(feats, axis=1)

    return np.concatenate([mean, std])


# ----------------------------------------------------
# Build feature table
# ----------------------------------------------------

def build_feature_rows() -> list[dict]:
    rows = []

    # 1) Bird references
    for path in sorted(REFERENCE_DIR.glob("*.wav")):
        y = load_audio(path)
        feat = extract_features(y)

        rows.append({
            "group": "bird_reference",
            "label": path.stem,
            "path": str(path),
            "feature": feat,
        })

    # 2) Office inputs + transformed outputs
    bench = pd.read_csv(BENCH_CSV)

    # To avoid duplicating office input points 3x, only add unique inputs once
    seen_inputs = set()

    for _, row in bench.iterrows():
        input_path = Path(row["input_file"])
        output_path = Path(row["output_file"])
        model_id = row["model_id"]

        if str(input_path) not in seen_inputs:
            y_in = load_audio(input_path)
            feat_in = extract_features(y_in)

            rows.append({
                "group": "office_input",
                "label": input_path.stem,
                "path": str(input_path),
                "feature": feat_in,
            })
            seen_inputs.add(str(input_path))

        y_out = load_audio(output_path)
        feat_out = extract_features(y_out)

        rows.append({
            "group": model_id,
            "label": output_path.stem,
            "path": str(output_path),
            "feature": feat_out,
        })

    return rows


# ----------------------------------------------------
# PCA plot
# ----------------------------------------------------

def main() -> int:
    print("[INFO] Building feature table...")
    rows = build_feature_rows()

    feats = np.stack([r["feature"] for r in rows], axis=0)

    print(f"[INFO] Total points: {len(rows)}")
    print(f"[INFO] Feature dim: {feats.shape[1]}")

    # Standardize features before PCA
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + EPS
    X = (feats - mean) / std

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plot_rows = []
    for r, xy in zip(rows, X_pca):
        plot_rows.append({
            "group": r["group"],
            "label": r["label"],
            "path": r["path"],
            "pc1": float(xy[0]),
            "pc2": float(xy[1]),
        })

    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved PCA coordinates: {OUT_CSV}")

    plt.figure(figsize=(9, 7))

    order = [
        "bird_reference",
        "office_input",
        "birds_dawnchorus_b2048_r48000_z8",
        "birds_motherbird_b2048_r48000_z16",
        "birds_pluma_b2048_r48000_z12",
    ]

    # Different alpha / size for readability
    style = {
        "bird_reference": {"alpha": 0.20, "s": 18},
        "office_input": {"alpha": 0.85, "s": 38},
        "birds_dawnchorus_b2048_r48000_z8": {"alpha": 0.75, "s": 38},
        "birds_motherbird_b2048_r48000_z16": {"alpha": 0.75, "s": 38},
        "birds_pluma_b2048_r48000_z12": {"alpha": 0.75, "s": 38},
    }

    for group in order:
        g = plot_df[plot_df["group"] == group]
        if len(g) == 0:
            continue

        plt.scatter(
            g["pc1"],
            g["pc2"],
            label=group,
            alpha=style[group]["alpha"],
            s=style[group]["s"],
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA Acoustic Map: Office, Bird Reference, and Transformed Outputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    print(f"[OK] Saved PCA plot: {OUT_PNG}")
    print(f"[INFO] Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
