#!/usr/bin/env python3

import argparse
import numpy as np
from numpy.linalg import norm, solve


def make_hilbert(n):
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)], dtype=np.float64)


def make_random(n, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n), dtype=np.float64)
    return A * scale


def compute_metrics(A, x, b, x_true):
    residual = norm(A.dot(x) - b)
    error = norm(x - x_true)
    return residual, error


def run_baseline(n, matrix_type="random"):
    if matrix_type == "hilbert":
        A = make_hilbert(n)
    else:
        A = make_random(n)

    x_true = np.ones(n, dtype=np.float64)
    b = A.dot(x_true)

    # FP64 ground truth
    x64 = solve(A, b)
    residual64, error64 = compute_metrics(A, x64, b, x_true)

    print(f"Baseline FP64 | n={n} | residual={residual64:.3e} | error={error64:.3e}")

    return {
        "n": n,
        "matrix_type": matrix_type,
        "residual64": residual64,
        "error64": error64,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CPU baseline experiments for mixed precision refinement.")
    parser.add_argument("--size", type=int, default=128, help="Matrix size")
    parser.add_argument("--matrix", choices=["random", "hilbert"], default="random", help="Matrix family")
    args = parser.parse_args()

    run_baseline(args.size, args.matrix)


if __name__ == "__main__":
    main()
