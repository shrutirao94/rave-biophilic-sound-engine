#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV = EXP_ROOT / "results" / "nature_distance_results.csv"

PLOT_DIR = EXP_ROOT / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    if not CSV.exists():
        raise RuntimeError(f"Missing results file: {CSV}")
    return pd.read_csv(CSV)


def plot_model_summary(df):
    summary = (
        df.groupby("model_id")["delta_nature_distance"]
        .agg(["mean", "std", "count"])
        .sort_values("mean")
    )

    plt.figure(figsize=(8,5))

    plt.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        capsize=6
    )

    plt.axhline(0)

    plt.ylabel("Δ Distance to Bird Distribution")
    plt.xlabel("Model")
    plt.title("Shift Toward Bird Acoustic Distribution")

    plt.xticks(rotation=25)

    plt.tight_layout()

    out = PLOT_DIR / "nature_distance_model_summary.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:", out)


def plot_distribution(df):

    models = df["model_id"].unique()

    data = [
        df[df["model_id"] == m]["delta_nature_distance"].values
        for m in models
    ]

    plt.figure(figsize=(8,5))

    plt.boxplot(
        data,
        labels=models
    )

    plt.axhline(0)

    plt.ylabel("Δ Distance to Bird Distribution")
    plt.xlabel("Model")
    plt.title("Distribution of Nature Similarity Shift")

    plt.xticks(rotation=25)

    plt.tight_layout()

    out = PLOT_DIR / "nature_distance_distribution.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:", out)


def plot_before_after_scatter(df):
    plt.figure(figsize=(7,7))
    for model, group in df.groupby("model_id"):
        plt.scatter(
            group["distance_input_to_birds"],
            group["distance_output_to_birds"],
            label=model,
            alpha=0.6
        )
    min_val = min(
        df["distance_input_to_birds"].min(),
        df["distance_output_to_birds"].min()
    )
    max_val = max(
        df["distance_input_to_birds"].max(),
        df["distance_output_to_birds"].max()
    )
    plt.plot(
        [min_val, max_val],
        [min_val, max_val]
    )
    plt.xlabel("Distance to Birds (Input)")
    plt.ylabel("Distance to Birds (Output)")
    plt.title("Input vs Output Distance to Bird Distribution")
    plt.legend()
    plt.tight_layout()
    out = PLOT_DIR / "nature_distance_scatter.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_model_points(df):

    summary = (
        df.groupby("model_id")
        .agg({
            "distance_input_to_birds": "mean",
            "distance_output_to_birds": "mean"
        })
    )

    plt.figure(figsize=(7,7))

    for model, row in summary.iterrows():

        x = row["distance_input_to_birds"]
        y = row["distance_output_to_birds"]

        plt.scatter(x, y, s=150)

        plt.text(
            x,
            y,
            model,
            fontsize=9,
            ha="left",
            va="bottom"
        )

    min_val = min(summary.min())
    max_val = max(summary.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("Distance to Birds (Input)")
    plt.ylabel("Distance to Birds (Output)")
    plt.title("Effect of Bird RAVE Models on Office Audio")

    plt.tight_layout()

    out = PLOT_DIR / "nature_distance_model_points.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print("Saved:", out)


def main():
    df = load_results()
    print("Loaded rows:", len(df))
    plot_model_summary(df)
    plot_distribution(df)
    plot_before_after_scatter(df)
    plot_model_points(df)
    print("\nAll plots saved to:", PLOT_DIR)


if __name__ == "__main__":
    main()
