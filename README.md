# Mixed-Precision Iterative Refinement Solver (GPU)

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://olebo.github.io/mixed-precision-iterative-refinement-gpu/)

A portfolio-grade GPU project demonstrating numerical maturity, mixed-precision design, and performance-aware implementation.

## Overview

This repository implements a solver for dense linear systems `Ax = b` using:

- low precision matrix factorization and solve (`FP16` / `FP32`)
- high precision residual computation and update (`FP64`)
- iterative refinement to recover `FP64`-level accuracy

The goal is to show that low precision compute can be used for speed, while high precision correction restores correctness.

## Why this project matters

This is not a toy. It is a signal of:

- algorithmic thinking
- numerical stability awareness
- GPU performance engineering
- comparison between low precision and mixed precision

## Architecture

Precision split:

- Matrix factorization: `FP16` or `FP32`
- Solve step: `FP16` / `FP32`
- Residual computation: `FP64`
- Update: `FP64`

## Project structure

```
mixed_precision_solver/
├── .github/
│   └── workflows/
├── docs/
│   └── index.md
├── include/
│   └── solver.h
├── python/
│   ├── experiments.py
│   └── plots.py
├── results/
│   ├── benchmarks.csv
│   └── plots.png
├── src/
│   ├── refinement.cu
│   ├── solver.cu
│   └── utils.cu
├── CMakeLists.txt
├── README.md
└── .gitignore
```

## Getting started

### Requirements

- CUDA-capable GPU
- CUDA Toolkit
- CMake
- Python 3.10+
- NumPy
- Matplotlib

### Build

```bash
mkdir -p build && cd build
cmake ..
make -j
```

### Python experiments

```bash
python3 python/experiments.py --size 512 --matrix hilbert
python3 python/plots.py --input results/benchmarks.csv
```

## Expected results

- `FP16` only: low accuracy or divergence
- `FP32` only: better, but not always enough for ill-conditioned systems
- Mixed precision iterative refinement: converges to `FP64`-level accuracy with much lower low-precision cost

## GitHub Pages

This repository includes GitHub Pages documentation served from `docs/`.
