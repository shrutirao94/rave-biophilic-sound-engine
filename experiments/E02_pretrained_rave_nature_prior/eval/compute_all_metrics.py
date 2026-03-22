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

BENCH = EXP_ROOT / "bench_eval_all.csv"
REFERENCE_ROOT = PROJECT_ROOT / "data/test/reference_eval"

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

    return float(np.corrcoef(env_a, env_b)[0, 1])


# ----------------------------------------------------
# Artifact score
# ----------------------------------------------------

def artifact_score(y):
    rms = librosa.feature.rms(y=y)[0]
    crest = np.max(np.abs(y)) / (np.mean(np.abs(y)) + EPS)
    spikes = np.sum(rms > np.mean(rms) + 3 * np.std(rms))
    return crest + spikes


# ----------------------------------------------------
# Reference distributions
# ----------------------------------------------------

def build_reference_distribution(reference_dir):
    feats = []
    for f in sorted(reference_dir.glob("*.wav")):
        y = load_audio(f)
        feats.append(extract_features(y))

    if len(feats) == 0:
        raise RuntimeError(f"No reference wav files found in {reference_dir}")

    X = np.vstack(feats)

    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    cov += np.eye(cov.shape[0]) * EPS
    inv_cov = np.linalg.inv(cov)

    return mu, inv_cov


def build_all_reference_distributions(reference_classes):
    ref_dists = {}
    for ref_class in sorted(reference_classes):
        ref_dir = REFERENCE_ROOT / ref_class

        if not ref_dir.exists():
            raise RuntimeError(f"Missing reference directory: {ref_dir}")

        print(f"Building reference distribution for: {ref_class}")
        ref_dists[ref_class] = build_reference_distribution(ref_dir)

    return ref_dists


# ----------------------------------------------------
# Plot functions
# ----------------------------------------------------

def plot_distance_shift(df):

    plt.figure(figsize=(8, 6))

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

    plt.xlabel("Distance to Reference (Input)")
    plt.ylabel("Distance to Reference (Output)")
    plt.title("Effect of Models on Input Audio")

    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distance_scatter.png")
    plt.close()


def plot_delta_distribution(df):
    plt.figure(figsize=(8, 6))
    df.boxplot(column="delta_distance", by="model_id")
    plt.axhline(0)
    plt.title("Shift Toward Reference Acoustic Distribution")
    plt.suptitle("")
    plt.ylabel("Δ Distance to Reference")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delta_distribution.png")
    plt.close()


def plot_improvement(summary):
    plt.figure(figsize=(8, 6))
    plt.bar(
        summary.index,
        summary["percent_improved"]
    )

    plt.ylabel("Percent of Segments Improved")
    plt.title("Segments Moving Closer to Their Reference Distribution")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "percent_improved.png")
    plt.close()


def plot_temporal(summary):
    plt.figure(figsize=(8, 6))
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

    bench = pd.read_csv(BENCH)

    reference_classes = sorted(bench["reference_class"].unique())
    print("Reference classes found in bench:", reference_classes)

    ref_dists = build_all_reference_distributions(reference_classes)

    rows = []

    for i, row in bench.iterrows():

        y_in = load_audio(row.input_file)
        y_out = load_audio(row.output_file)

        f_in = extract_features(y_in)
        f_out = extract_features(y_out)

        ref_class = row.reference_class
        mu_ref, cov_inv = ref_dists[ref_class]

        d_in = mahalanobis(f_in, mu_ref, cov_inv)
        d_out = mahalanobis(f_out, mu_ref, cov_inv)

        delta = d_out - d_in
        corr = envelope_corr(y_in, y_out)
        art = artifact_score(y_out)

        rows.append({
            "model_id": row.model_id,
            "reference_class": ref_class,
            "input_file": row.input_file,
            "output_file": row.output_file,
            "distance_input": d_in,
            "distance_output": d_out,
            "delta_distance": delta,
            "envelope_correlation": corr,
            "artifact_score": art
        })

        if (i + 1) % 20 == 0:
            print(i + 1, "segments processed")

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)

    print("Saved results:", RESULT_CSV)

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    summary = df.groupby(["reference_class", "model_id"]).agg({
        "delta_distance": ["mean", "std"],
        "envelope_correlation": ["mean", "std"],
        "artifact_score": ["mean", "std"]
    })

    summary.columns = ["_".join(c) for c in summary.columns]

    improvement = df.groupby(["reference_class", "model_id"]).apply(
        lambda g: np.mean(g["delta_distance"] < 0)
    )

    summary["percent_improved"] = improvement * 100
    summary["count"] = df.groupby(["reference_class", "model_id"]).size()

    summary.to_csv(SUMMARY_CSV)

    print("Saved summary:", SUMMARY_CSV)

    # ----------------------------------------------------
    # Plots
    # ----------------------------------------------------

    print("Generating plots...")

    plot_distance_shift(df)
    plot_delta_distribution(df)
    plot_improvement(summary.reset_index().set_index("model_id"))
    plot_temporal(summary.reset_index().set_index("model_id"))

    print("Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
