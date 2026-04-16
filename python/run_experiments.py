#!/usr/bin/env python3

import csv
import numpy as np
from numpy.linalg import solve, norm

from baseline import make_random, make_hilbert, compute_metrics


def run_experiments(sizes, K=50, matrix_type="random"):
    rows = []

    for n in sizes:
        for seed in range(K):
            if matrix_type == "hilbert":
                A = make_hilbert(n)
            else:
                A = make_random(n, seed=seed)

            x_true = np.ones(n)
            b = A.dot(x_true)

            x = solve(A, b)
            r, e = compute_metrics(A, x, b, x_true)

            rows.append([n, matrix_type, seed, r, e])
    return rows

    


if __name__ == "__main__":
    out_file="results/baseline_results.csv"
    sizes = [64, 128, 256, 512, 1024]
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "matrix", "seed", "residual", "error"])
        rows_r = run_experiments(sizes, K=50, matrix_type="random")
        writer.writerows(rows_r)
        rows_h = run_experiments(sizes, K=1,  matrix_type="hilbert")

    print(f"Saved {len(rows_r)+len(rows_h)} rows to {out_file}")