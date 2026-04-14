---
title: Mathematical Foundations
nav_order: 2
---

# Mixed-Precision Linear Solver (CUDA) — Mathematical Documentation

## 1. Problem Statement

We consider the numerical solution of a linear system:

$$
A x = b, \quad A \in \mathbb{R}^{n \times n}, \; x,b \in \mathbb{R}^n
$$

where:
- $A$ is typically dense and non-singular
- $b$ is a given right-hand side
- $x$ is the unknown solution

The goal is to compute an accurate solution efficiently using **mixed precision arithmetic** on GPUs.

---

## 2. Motivation for Mixed Precision

### 2.1 Precision Types
- **Single precision (FP32)**: fast on GPUs, lower accuracy (~7 decimal digits)
- **Double precision (FP64)**: slower, higher accuracy (~16 decimal digits)

### 2.2 Key Idea

We:
1. Solve the system approximately in **low precision (FP32)**
2. Iteratively refine the solution in **high precision (FP64)**

This yields:
- Performance close to FP32
- Accuracy close to FP64

---

## 3. Algorithm Overview

The code implements a **mixed-precision iterative refinement scheme**.

### Step 1: Initial Solve (Low Precision)

Solve:
$$
A^{(f)} x^{(0)} = b^{(f)}
$$

where $A^{(f)}$, $b^{(f)}$ are FP32 versions of $A$, $b$.

This is done on the GPU via:

```
gpuSolve(...)
```

Mathematically:
$$
x^{(0)} \approx A^{-1} b
$$

---

### Step 2: Iterative Refinement Loop

For $k = 0,1,2,\dots$:

#### 3.1 Residual Computation (High Precision)

$$
r^{(k)} = b - A x^{(k)}
$$

- Computed in FP64 to avoid loss of significance
- Measures current error

---

#### 3.2 Correction Solve (Low Precision)

Solve:
$$
A \, \delta x^{(k)} = r^{(k)}
$$

BUT in practice:
- $A$ and $r^{(k)}$ are cast to FP32
- Solve is performed on GPU

---

#### 3.3 Update Step

$$
x^{(k+1)} = x^{(k)} + \delta x^{(k)}
$$

---

#### 3.4 Convergence Check

Stop when:
$$
\|r^{(k)}\| \leq \varepsilon
$$

or after `maxIter` iterations.

---

## 4. Mathematical Foundations

### 4.1 Error Propagation

Let the true solution be $x^*$. Define the error:

$$
e^{(k)} = x^* - x^{(k)}
$$

Then:
$$
r^{(k)} = A e^{(k)}
$$

Solving:
$$
A \delta x^{(k)} = r^{(k)}
\Rightarrow \delta x^{(k)} = e^{(k)}
$$

Thus ideally:
$$
x^{(k+1)} = x^{(k)} + e^{(k)} = x^*
$$

In exact arithmetic, convergence happens in **one step**.

---

### 4.2 Finite Precision Effects

In floating-point arithmetic:
- Residual is computed accurately (FP64)
- Correction solve is approximate (FP32)

We effectively solve:
$$
(A + \Delta A) \delta x^{(k)} = r^{(k)}
$$

This leads to a contraction:
$$
\|e^{(k+1)}\| \leq C \|e^{(k)}\|
$$

with convergence if:
$$
\kappa(A) \cdot u_{\text{low}} < 1
$$

where:
- $\kappa(A)$: condition number
- $u_{\text{low}}$: machine precision of FP32

---

### 4.3 Convergence Condition

For successful refinement:

$$
\kappa(A) \lesssim \frac{1}{u_{\text{float}}} \approx 10^7
$$

If the system is too ill-conditioned, refinement may fail.

---

## 5. Code Structure

### 5.1 Namespace

```
namespace mixed_precision
```

Encapsulates all solver functionality.

---

### 5.2 Lifecycle Functions

#### `initializeCuda()`
- Initializes CUDA runtime
- Likely sets up cuBLAS / cuSOLVER handles

#### `shutdownCuda()`
- Releases GPU resources

---

### 5.3 Core Solver

#### `gpuSolve(const float* d_A, const float* d_b, float* d_x, int n)`

- Inputs:
  - `d_A`: matrix in device memory (FP32)
  - `d_b`: RHS vector (FP32)
- Output:
  - `d_x`: solution (FP32)

Mathematically:
$$
\text{solve } A x = b
$$

Implementation likely uses:
- LU decomposition via cuSOLVER
- or QR factorization

---

### 5.4 Iterative Refinement

#### `refineSolution(const double* A, const double* b, double* x, int n, int maxIter)`

This is the **main algorithmic component**.

Responsibilities:
1. Convert inputs to FP32 for GPU solve
2. Compute residual in FP64
3. Call `gpuSolve` for correction
4. Update solution

---

## 6. Data Flow Between Precisions

| Stage | Precision | Location |
|------|----------|----------|
| Initial solve | FP32 | GPU |
| Residual computation | FP64 | CPU (or GPU FP64) |
| Correction solve | FP32 | GPU |
| Update | FP64 | CPU |

---

## 7. Numerical Stability Considerations

### 7.1 Conditioning
- Poorly conditioned matrices degrade convergence
- Preconditioning may be required

### 7.2 Scaling
- Normalize rows/columns to reduce condition number

### 7.3 Pivoting
- LU factorization must use partial pivoting

---

## 8. Performance Considerations

### 8.1 Why GPUs?
- FP32 throughput is significantly higher
- Memory bandwidth optimized for dense linear algebra

### 8.2 Bottlenecks
- CPU–GPU transfers
- Residual computation in FP64

---

## 9. Error Handling

### CUDA Macro

```
CUDA_CHECK(call)
```

Wraps CUDA calls and reports:
- File
- Line number
- Error string

---

### cuBLAS Macro

```
CUBLAS_CHECK(call)
```

Handles non-CUDA return types.

---

## 10. Summary of the Mathematical Idea

The entire solver is based on the identity:

$$
x = x + A^{-1}(b - A x)
$$

which is implemented iteratively using:
- **Fast approximate solves (FP32)**
- **Accurate residuals (FP64)**

This is a classic **HPC technique** used in:
- Numerical linear algebra libraries
- Scientific computing
- Machine learning solvers

---

## 11. Extensions

Possible improvements:
- Preconditioned iterative refinement
- GMRES-based correction instead of direct solve
- Fully GPU-based FP64 residual computation
- Batched solves for multiple RHS

---

## 12. Key Insight

Mixed precision works because:

> The expensive part (matrix factorization) is done in low precision,
> while the accuracy-critical part (residual) is done in high precision.

This exploits modern GPU architectures where FP32 is extremely fast.

---

**End of Documentation**

