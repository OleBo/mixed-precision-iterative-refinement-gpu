---
title: Benchmark Experiments
nav_order: 3
---

# Benchmark Experiment Documentation — CPU Baseline (FP64 Ground Truth)

## 1. Objective

Establish a **reliable numerical ground truth** for evaluating the mixed-precision GPU solver.

This baseline:
- Runs entirely on CPU
- Uses **double precision (FP64)**
- Provides reference values for:
  - Residual: $ \|Ax - b\| $
  - Solution error: $ \|x - x_{true}\| $

These metrics serve as the **gold standard** for all subsequent comparisons.

---

## 2. Experimental Design

### 2.1 Linear System

We solve:

$$
A x = b
$$

where:
- $A \in \mathbb{R}^{n \times n}$
- $x_{true} = \mathbf{1}$ (vector of ones)
- $b = A x_{true}$

This construction ensures:
- The exact solution is known
- Error can be measured explicitly

---

### 2.2 Matrix Families

#### 1. Random Gaussian Matrix

$$
A_{ij} \sim \mathcal{N}(0, 1)
$$

Properties:
- Well-conditioned (typically)
- Represents "easy" numerical cases

---

#### 2. Hilbert Matrix

$$
A_{ij} = \frac{1}{i + j + 1}
$$

Properties:
- Extremely ill-conditioned
- Classical stress test for numerical solvers

---

## 3. Metrics

### 3.1 Residual

$$
\|Ax - b\|_2
$$

Interpretation:
- Measures how well the computed solution satisfies the equation
- Small residual ⇒ equation nearly satisfied

---

### 3.2 Solution Error

$$
\|x - x_{true}\|_2
$$

Interpretation:
- Measures distance to the exact solution
- Captures accumulated numerical error

---

### 3.3 Important Distinction

A solver may have:
- **Low residual but high error** (ill-conditioned case)
- This highlights the role of the condition number:

$$
\|x - x^*\| \leq \kappa(A) \cdot \frac{\|r\|}{\|A\|}
$$

---

## 4. Implementation (Python / NumPy)

The following script defines the CPU baseline experiment:

```python
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
```

---

## 5. Expected Behavior

### 5.1 Random Matrices

- Residual: $ \approx 10^{-14} $ to $10^{-16}$
- Error: $ \approx 10^{-14} $

Interpretation:
- Near machine precision accuracy
- Well-conditioned system

---

### 5.2 Hilbert Matrices

- Residual: small
- Error: **large** (may grow rapidly with $n$)

Interpretation:
- Ill-conditioning amplifies numerical errors
- Demonstrates limits of floating-point arithmetic

---

## 6. Experimental Protocol

### Step 1: Run Baseline

Example:

```bash
python baseline.py --size 128 --matrix random
python baseline.py --size 128 --matrix hilbert
```

---

### Step 2: Sweep Problem Sizes

Recommended sizes:

$$
n \in \{64, 128, 256, 512, 1024\}
$$

---

### Step 3: Record Results

Store results in a table:

| n | matrix | residual (FP64) | error (FP64) |
|---|--------|-----------------|--------------|
| 128 | random | ... | ... |
| 128 | hilbert | ... | ... |

---

## 7. Role in Mixed-Precision Pipeline

This baseline is used to:

1. Validate GPU solver correctness
2. Compare:
   - FP32 solve
   - Mixed-precision refinement
3. Quantify accuracy loss vs recovery

---

## 8. Key Insight

> The CPU FP64 solution is not just a reference — it defines what "correct" means numerically.

All GPU results should be evaluated relative to this baseline.

---

## 9. Extensions

Possible improvements:

- Add condition number estimation:
  $$
  \kappa(A) = \|A\| \|A^{-1}\|
  $$

- Add timing measurements
- Export results to CSV for plotting
- Compare different matrix scalings

---

## 10. Summary

This experiment establishes:

- A **numerically trustworthy baseline**
- A framework for **quantitative comparison**
- A foundation for evaluating **mixed-precision algorithms**

---

## 10. Statistical Significance for Random Matrices

### Motivation

Random matrices introduce **stochastic variability**:
- Each run produces a different matrix (A)
- Metrics (residual, error) become random variables

Therefore, a single run is **not sufficient** to draw conclusions.

---

### 10.1 Experimental Design

For each configuration (n, matrix type):

- Run experiment K times with different seeds
- Collect:
  r_i = ||A_i x_i - b_i||,  e_i = ||x_i - x_true||

Typical choice:
K = 20 to 100

---

### 10.2 Sample Statistics

Compute:

Mean:
 r_bar = (1/K) sum r_i
 e_bar = (1/K) sum e_i

Standard deviation:
 sigma_r = sqrt( (1/(K-1)) sum (r_i - r_bar)^2 )

---

### 10.3 Confidence Intervals

Assuming approximate normality:

 r_bar ± t * sigma_r / sqrt(K)

For 95% confidence:
 t ≈ 2

---

### 10.4 Comparing Methods

Use paired experiments:
- Same matrix A_i for all methods

Analyze:
 Delta e_i = e_i(method1) - e_i(method2)

Apply:
- Paired t-test

---

### 10.5 Practical Implementation

```python
def run_multiple(n, K=50):
    results = []
    for seed in range(K):
        A = make_random(n, seed=seed)
        x_true = np.ones(n)
        b = A.dot(x_true)

        x = solve(A, b)
        r, e = compute_metrics(A, x, b, x_true)
        results.append((r, e))

    results = np.array(results)
    mean = results.mean(axis=0)
    std = results.std(axis=0, ddof=1)

    return mean, std
```

---

### 10.6 Key Insight

Without statistical analysis, performance claims on random matrices are unreliable.

---

## 11. Summary

This benchmark framework now provides:

- Deterministic ground truth (Hilbert)
- Statistical robustness (random matrices)
- Quantitative metrics for solver evaluation

---

**Status**: Baseline + Statistical Validation — COMPLETE


---

## 12. Full Experimental Pipeline (CSV + Plots)

This section upgrades the baseline into a **reproducible benchmarking pipeline** with:
- Multiple runs (statistics)
- CSV logging
- Automated plotting (matplotlib)

---

### 12.1 Directory Structure

```
experiments/
  baseline.py
  run_experiments.py
  plot_results.py
  results/
    baseline_results.csv
    summary_results.csv
    plots/
```

---

### 12.2 CSV Logging Schema

Each run produces one row:

| n | matrix | seed | residual | error |
|---|--------|------|----------|-------|

---

### 12.3 Batch Experiment Script

```python
# run_experiments.py
import csv
import numpy as np
from numpy.linalg import solve, norm

from baseline import make_random, make_hilbert, compute_metrics


def run_experiments(sizes, K=50, matrix_type="random", out_file="results/baseline_results.csv"):
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

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "matrix", "seed", "residual", "error"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    sizes = [64, 128, 256, 512, 1024]
    run_experiments(sizes, K=50, matrix_type="random")
```

---

### 12.4 Aggregation Script (Statistics)

```python
# aggregate_results.py
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
```

---

### 12.5 Plotting Script (matplotlib)

```python
# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt


def plot(file="results/summary_results.csv"):
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

        plt.savefig(f"results/plots/{metric}.png")
        plt.close()


if __name__ == "__main__":
    plot()
```

---

### 12.6 Recommended Workflow

```bash
# 1. Run experiments
python run_experiments.py

# 2. Aggregate statistics
python aggregate_results.py

# 3. Generate plots
python plot_results.py
```

---

### 12.7 Expected Plots

You should obtain:

- Residual vs n (log-log)
- Error vs n (log-log)
- Error bars showing variability

Key observations:
- Random matrices: stable, low variance
- Hilbert: exploding error with n

---

### 12.8 Extension: GPU Comparison

Add columns:
- error_fp32
- error_mixed

Then plot all curves together to show:

- FP32 degradation
- Mixed precision recovery

---

### 12.9 Key Insight

> A benchmark is only credible if it is reproducible, statistical, and visual.

This pipeline satisfies all three.

---

## 13. GPU Integration (FP32 + Mixed Precision)

This section extends the pipeline to include:
- FP64 CPU baseline
- FP32 GPU solve
- Mixed-precision iterative refinement (GPU + FP64 residual)

---

### 13.1 Python–CUDA Bridge

Assume you expose your CUDA code via a shared library:

```bash
nvcc -O3 -Xcompiler -fPIC -shared solver.cu refinement.cu -o libmpsolver.so -lcublas -lcusolver
```

---

### 13.2 ctypes Wrapper

```python
# gpu_wrapper.py
import numpy as np
import ctypes

lib = ctypes.CDLL("./libmpsolver.so")

lib.gpuSolve.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

lib.refineSolution.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
]


def gpu_solve(A, b):
    n = A.shape[0]

    A32 = A.astype(np.float32).copy(order="C")
    b32 = b.astype(np.float32).copy(order="C")
    x32 = np.zeros_like(b32)

    lib.gpuSolve(
        A32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )

    return x32.astype(np.float64)


def gpu_refine(A, b, max_iter=10):
    n = A.shape[0]

    A64 = A.astype(np.float64).copy(order="C")
    b64 = b.astype(np.float64).copy(order="C")
    x64 = np.zeros_like(b64)

    lib.refineSolution(
        A64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        max_iter,
    )

    return x64
```

---

### 13.3 Extended Experiment Script

```python
# run_full_experiments.py
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
```

---

### 13.4 Aggregation (Comparison)

```python
# aggregate_full.py
import pandas as pd


def aggregate(file="results/full_results.csv"):
    df = pd.read_csv(file)

    return df.groupby("n").agg({
        "err64": ["mean","std"],
        "err32": ["mean","std"],
        "err_mixed": ["mean","std"],
    })
```

---

### 13.5 Money Plot (Core Result)

```python
# plot_full.py
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
```

---

### 13.6 Expected Outcome (Critical Insight)

You should observe:

- FP64: flat near machine precision
- FP32: error grows significantly
- Mixed precision: tracks FP64 closely

This demonstrates:

> Mixed precision recovers FP64 accuracy at near FP32 cost.

---

## 14. Final Result

You now have a **publication-grade benchmark**:

- Deterministic baseline
- Statistical validation
- GPU acceleration
- Clear comparison visualization

---

**Status**: Full GPU-Integrated Benchmark — COMPLETE

