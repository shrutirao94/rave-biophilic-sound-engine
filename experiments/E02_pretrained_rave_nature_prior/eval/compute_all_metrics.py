#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BENCH = EXP_ROOT / "bench_eval.csv"
REFERENCE_DIR = PROJECT_ROOT / "data/test/reference_eval/birds"

RESULT_DIR = EXP_ROOT / "results"
FIG_DIR = EXP_ROOT / "figures"

RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

RESULT_CSV = RESULT_DIR / "evaluation_results.csv"
SUMMARY_CSV = RESULT_DIR / "evaluation_summary.csv"

SR = 48000
EPS = 1e-6


# ----------------------------------------------------
# Audio loading
# ----------------------------------------------------

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


# ----------------------------------------------------
# Feature extraction
# ----------------------------------------------------

def extract_features(y):

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
        onset.reshape(1, -1)
    ])

    mean = np.mean(feats, axis=1)
    std = np.std(feats, axis=1)

    return np.concatenate([mean, std])


# ----------------------------------------------------
# Envelope similarity
# ----------------------------------------------------

def envelope_corr(a, b):

    env_a = librosa.feature.rms(y=a)[0]
    env_b = librosa.feature.rms(y=b)[0]

    n = min(len(env_a), len(env_b))
    env_a = env_a[:n]
    env_b = env_b[:n]

    if np.std(env_a) < EPS or np.std(env_b) < EPS:
        return np.nan

    return float(np.corrcoef(env_a, env_b)[0,1])


# ----------------------------------------------------
# Artifact score
# ----------------------------------------------------

def artifact_score(y):

    rms = librosa.feature.rms(y=y)[0]

    crest = np.max(np.abs(y)) / (np.mean(np.abs(y)) + EPS)

    spikes = np.sum(rms > np.mean(rms) + 3*np.std(rms))

    return crest + spikes


# ----------------------------------------------------
# Reference distribution
# ----------------------------------------------------

def build_reference_distribution():

    feats = []

    for f in sorted(REFERENCE_DIR.glob("*.wav")):

        y = load_audio(f)
        feats.append(extract_features(y))

    X = np.vstack(feats)

    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    cov += np.eye(cov.shape[0]) * EPS

    inv_cov = np.linalg.inv(cov)

    return mu, inv_cov


# ----------------------------------------------------
# Plot functions
# ----------------------------------------------------

def plot_distance_shift(df):

    plt.figure(figsize=(8,6))

    for model, g in df.groupby("model_id"):
        plt.scatter(
            g["distance_input"],
            g["distance_output"],
            label=model,
            alpha=0.7
        )

    lims = [
        min(df.distance_input.min(), df.distance_output.min()),
        max(df.distance_input.max(), df.distance_output.max())
    ]

    plt.plot(lims, lims, '--')

    plt.xlabel("Distance to Birds (Input)")
    plt.ylabel("Distance to Birds (Output)")
    plt.title("Effect of Bird RAVE Models on Office Audio")

    plt.legend()

    plt.tight_layout()

    plt.savefig(FIG_DIR / "distance_scatter.png")
    plt.close()


def plot_delta_distribution(df):

    plt.figure(figsize=(8,6))

    df.boxplot(column="delta_distance", by="model_id")

    plt.axhline(0)

    plt.title("Shift Toward Bird Acoustic Distribution")
    plt.suptitle("")

    plt.ylabel("Δ Distance to Bird Distribution")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(FIG_DIR / "delta_distribution.png")
    plt.close()


def plot_improvement(summary):

    plt.figure(figsize=(8,6))

    plt.bar(
        summary.index,
        summary["percent_improved"]
    )

    plt.ylabel("Percent of Segments Improved")
    plt.title("Segments Moving Closer to Bird Distribution")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(FIG_DIR / "percent_improved.png")
    plt.close()


def plot_temporal(summary):

    plt.figure(figsize=(8,6))

    plt.bar(
        summary.index,
        summary["envelope_correlation_mean"]
    )

    plt.ylabel("Envelope Correlation")
    plt.title("Temporal Structure Preservation")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(FIG_DIR / "temporal_preservation.png")
    plt.close()


# ----------------------------------------------------
# Main evaluation
# ----------------------------------------------------

def main():

    print("Building bird reference distribution...")

    mu_ref, cov_inv = build_reference_distribution()

    bench = pd.read_csv(BENCH)

    rows = []

    for i, row in bench.iterrows():

        y_in = load_audio(row.input_file)
        y_out = load_audio(row.output_file)

        f_in = extract_features(y_in)
        f_out = extract_features(y_out)

        d_in = mahalanobis(f_in, mu_ref, cov_inv)
        d_out = mahalanobis(f_out, mu_ref, cov_inv)

        delta = d_out - d_in

        corr = envelope_corr(y_in, y_out)

        art = artifact_score(y_out)

        rows.append({

            "model_id": row.model_id,

            "distance_input": d_in,
            "distance_output": d_out,

            "delta_distance": delta,

            "envelope_correlation": corr,

            "artifact_score": art
        })

        if (i+1) % 20 == 0:
            print(i+1, "segments processed")

    df = pd.DataFrame(rows)

    df.to_csv(RESULT_CSV, index=False)

    print("Saved results:", RESULT_CSV)


# ----------------------------------------------------
# Summary
# ----------------------------------------------------

    summary = df.groupby("model_id").agg({

        "delta_distance": ["mean","std"],
        "envelope_correlation": ["mean","std"],
        "artifact_score": ["mean","std"]

    })

    summary.columns = ["_".join(c) for c in summary.columns]

    improvement = df.groupby("model_id").apply(
        lambda g: np.mean(g["delta_distance"] < 0)
    )

    summary["percent_improved"] = improvement * 100
    summary["count"] = df.groupby("model_id").size()

    summary.to_csv(SUMMARY_CSV)

    print("Saved summary:", SUMMARY_CSV)


# ----------------------------------------------------
# Plots
# ----------------------------------------------------

    print("Generating plots...")

    plot_distance_shift(df)
    plot_delta_distribution(df)
    plot_improvement(summary)
    plot_temporal(summary)

    print("Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
