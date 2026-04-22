#!/usr/bin/env python3

import csv
import numpy as np
from numpy.linalg import solve

from baseline import make_random, compute_metrics
from gpu_wrapper import gpu_solve, gpu_refine


def run(sizes, K=30, out_file="results/full_results.csv"):
    rows = []

    for n in sizes:
        for seed in range(K):
            A = make_random(n, seed=seed)
            x_true = np.ones(n)
            b = A.dot(x_true)

            # FP64 baseline
            x64 = solve(A, b)
            r64, e64 = compute_metrics(A, x64, b, x_true)

            # FP32 GPU
            x32 = gpu_solve(A, b)
            r32, e32 = compute_metrics(A, x32, b, x_true)

            # Mixed precision
            xm = gpu_refine(A, b)
            rm, em = compute_metrics(A, xm, b, x_true)

            rows.append([n, seed, r64, e64, r32, e32, rm, em])

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n","seed",
            "res64","err64",
            "res32","err32",
            "res_mixed","err_mixed"
        ])
        writer.writerows(rows)
        
if __name__ == "__main__":
    sizes = [64, 128, 256, 512, 1024]
    run(sizes)