#!/usr/bin/env python3

import pandas as pd


def aggregate(in_file="results/baseline_results.csv", out_file="results/summary_results.csv"):
    df = pd.read_csv(in_file)

    grouped = df.groupby(["n", "matrix"]).agg(
        residual_mean=("residual", "mean"),
        residual_std=("residual", "std"),
        error_mean=("error", "mean"),
        error_std=("error", "std"),
    ).reset_index()

    grouped.to_csv(out_file, index=False)
    print(grouped)


if __name__ == "__main__":
    aggregate()
