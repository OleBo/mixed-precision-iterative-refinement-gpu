# Mixed-Precision Iterative Refinement Solver (GPU)

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://olebo.github.io/mixed-precision-iterative-refinement-gpu/) 
[![CI](https://github.com/OleBo/mixed-precision-iterative-refinement-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/OleBo/mixed-precision-iterative-refinement-gpu/actions/workflows/build.yml)


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

### Docker

Build and run the solver in a containerized environment with GPU support:

```bash
# Build the image
docker build -t mixed-precision-solver .

# Run with GPU support
docker run --gpus all -it mixed-precision-solver
```

Or use Docker Compose:

```bash
# Build and start the container
docker-compose up -d

# Execute commands
docker-compose exec mixed-precision-solver bash
```

The Dockerfile uses a multi-stage build process:
- **Builder stage**: Compiles CUDA code on `nvidia/cuda:12.2.2-devel`
- **Runtime stage**: Runs on `nvidia/cuda:12.2.2-runtime` with Python support

### Python experiments

```bash
python3 python/experiments.py --size 512 --matrix hilbert
python3 python/plots.py --input results/benchmarks.csv
```

## Expected results

- `FP16` only: low accuracy or divergence
- `FP32` only: better, but not always enough for ill-conditioned systems
- Mixed precision iterative refinement: converges to `FP64`-level accuracy with much lower low-precision cost

## Continuous Integration

This project uses GitHub Actions for automated building and testing:

- **Workflow**: `.github/workflows/build.yml`
- **Triggers**: Push to `main`/`develop` branches and pull requests
- **Jobs**:
  - Build Docker image and push to GitHub Container Registry (GHCR)
  - Run solver in container to verify functionality
  - Cache Docker layers for faster builds

View workflow runs in the [Actions](./../../actions) tab.

## GitHub Pages

This repository includes GitHub Pages documentation served from `docs/`.
