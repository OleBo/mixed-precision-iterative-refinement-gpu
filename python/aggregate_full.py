#!/usr/bin/env python3

import pandas as pd


def aggregate(file="results/full_results.csv"):
    df = pd.read_csv(file)

    return df.groupby("n").agg({
        "err64": ["mean","std"],
        "err32": ["mean","std"],
        "err_mixed": ["mean","std"],
    })

if __name__ == "__main__":
    grouped = aggregate()
    grouped.to_csv("results/aggregate_full_results.csv", index=False)
    print(grouped)

