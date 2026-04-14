# Mixed-Precision Iterative Refinement on GPUs

[![Build](https://img.shields.io/github/actions/workflow/status/OleBo/mixed-precision-iterative-refinement-gpu/build.yml?branch=main)](https://github.com/OleBo/mixed-precision-iterative-refinement-gpu/actions)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/github/license/OleBo/mixed-precision-iterative-refinement-gpu)](https://github.com/OleBo/mixed-precision-iterative-refinement-gpu/blob/main/LICENSE)

---

Welcome to this project on **mixed-precision linear solvers using CUDA and modern HPC techniques**.

This repository demonstrates how to combine:
- **FP32 performance on GPUs**
- with **FP64 accuracy via iterative refinement**

The result is a solver that achieves **high performance without sacrificing numerical precision** — a key idea in modern scientific computing and AI workloads.

---

## 🔍 What You’ll Find Here

This project is structured around three core components:

### 1. Mathematical Foundations
A detailed explanation of the algorithm:
- Linear system formulation
- Iterative refinement theory
- Convergence conditions and numerical stability

👉 [Mixed Precision Solver Documentation](./Mixed%20Precision%20Solver%20Documentation.md)

---

### 2. Experimental Evaluation
A full benchmarking pipeline including:
- CPU baseline (FP64 ground truth)
- Statistical analysis (random matrices)
- CSV logging and reproducible experiments
- Visualization with matplotlib
- GPU vs FP32 vs mixed-precision comparison (“money plot”)

👉 [Benchmark Experiment Documentation](./Benchmark%20Experiment%20Documentation.md)

---

### 3. CUDA Implementation
A deep dive into the GPU implementation:
- Memory management and data movement
- Parallelization strategy (threads, blocks, kernels)
- Use of cuBLAS / cuSOLVER
- Memory hierarchy and tiling
- Template-based design patterns

👉 [CUDA Solver Implementation Documentation](./CUDA%20Solver%20Implementation%20Documentation.md)

---

## 🚀 Project Repository

Full source code is available here:

👉 https://github.com/OleBo/mixed-precision-iterative-refinement-gpu

---

## 💡 Why This Matters

Mixed-precision methods are widely used in:
- High-performance computing (HPC)
- Scientific simulations
- Machine learning and AI

This project shows how to:
- **bridge theory and implementation**
- **leverage GPU architectures effectively**
- **build reproducible numerical experiments**

---

## 📌 Key Takeaway

> You don’t need full precision everywhere — you just need it where it matters.

---

## 👤 Author

This project is part of a portfolio exploring:
- Numerical linear algebra
- GPU programming (CUDA)
- High-performance computing systems