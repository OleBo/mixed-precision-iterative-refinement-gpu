#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt


def plot(file="results/summary_baseline_results.csv"):
    df = pd.read_csv(file)

    for metric in ["residual", "error"]:
        plt.figure()

        for matrix in df["matrix"].unique():
            sub = df[df["matrix"] == matrix]

            x = sub["n"]
            y = sub[f"{metric}_mean"]
            err = sub[f"{metric}_std"]

            plt.errorbar(x, y, yerr=err, label=matrix, marker='o')

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Matrix size n")
        plt.ylabel(metric)
        plt.title(f"{metric} vs size")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"results/plots/baseline_{metric}.png")
        plt.close()


if __name__ == "__main__":
    plot()