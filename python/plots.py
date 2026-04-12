#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path, image_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(df["iteration"], df["residual"], marker="o", label="Residual")
    ax[0].set_ylabel("Residual")
    ax[0].set_yscale("log")
    ax[0].legend()

    ax[1].plot(df["iteration"], df["error"], marker="o", label="Error", color="tab:orange")
    ax[1].set_ylabel("Error")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Iteration")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(image_path)
    print(f"Saved plot to {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot mixed precision solver results.")
    parser.add_argument("--input", default="results/benchmarks.csv", help="Input CSV file")
    parser.add_argument("--output", default="results/plots.png", help="Output image file")
    args = parser.parse_args()
    plot_metrics(args.input, args.output)


if __name__ == "__main__":
    main()
