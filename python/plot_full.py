#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt


def plot(file="results/full_results.csv"):
    df = pd.read_csv(file)
    grouped = df.groupby("n").mean().reset_index()

    plt.figure()

    plt.plot(grouped["n"], grouped["err64"], marker='o', label="FP64")
    plt.plot(grouped["n"], grouped["err32"], marker='o', label="FP32 GPU")
    plt.plot(grouped["n"], grouped["err_mixed"], marker='o', label="Mixed Precision")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.title("Solver Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/plots/comparison.png")
    plt.close()