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

> **Remark: Hardware Acceleration**
> Modern GPU architectures (e.g., NVIDIA Ampere/Hopper) utilize **Tensor Cores** to [significantly accelerate](https://www.google.com/url?sa=i&source=web&rct=j&url=https://pmc.ncbi.nlm.nih.gov/articles/PMC7735315/&ved=2ahUKEwiFwPq_7vGTAxWXSPEDHdTiHZoQy_kOegYIAQgMEAM&opi=89978449&cd&psig=AOvVaw2tTWhaKJZnGeino5eM4wgU&ust=1776411456484000) low-precision matrix math. While this provides a massive throughput boost for the "Correction Solve" phase, it requires careful monitoring of the matrix condition number to ensure convergence.

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

### 4.1 Error Propagation and Ideal Iteration
Let $x^*$ be the exact solution to $Ax = b$. We define the error at iteration $k$ as:
$$e^{(k)} = x^* - x^{(k)}$$

The **residual** $r^{(k)}$, which represents the remaining error in the system, is computed as:
$$r^{(k)} = b - Ax^{(k)} = A(x^* - x^{(k)}) = Ae^{(k)}$$

By solving the system for the correction term $\delta x^{(k)}$:
$$A \delta x^{(k)} = r^{(k)} \implies \delta x^{(k)} = e^{(k)}$$

In **exact arithmetic**, the solution is updated as $x^{(k+1)} = x^{(k)} + \delta x^{(k)}$, which yields the true solution $x^*$ in a single step.

---

### 4.2 Finite Precision Effects
In mixed-precision arithmetic, errors introduced during computation prevent one-step convergence:

1.  **High-Precision Residual:** $r^{(k)}$ is computed in **FP64**. This minimizes the impact of *cancellation error*, ensuring the residual reflects the remaining error rather than rounding noise.
2.  **Approximate Correction:** The correction $\delta x^{(k)}$ is solved in **FP32**. We effectively solve a perturbed system:
    $$(A + \delta A^{(k)}) \widehat{\delta x}^{(k)} = r^{(k)}$$
    where $\delta A^{(k)}$ represents the backward error of the low-precision solver.

Because $\widehat{\delta x}^{(k)}$ is an approximation, the update $x^{(k+1)} = x^{(k)} + \widehat{\delta x}^{(k)}$ behaves as a **contraction mapping**:
$$\|e^{(k+1)}\| \leq C \cdot \|e^{(k)}\|, \quad \text{where } C < 1$$

---

### 4.3 Convergence Condition
The refinement converges toward FP64 accuracy if the error introduced by the low-precision solve does not exceed the information provided by the residual:
$$\kappa(A) \cdot u_{\text{low}} < 1$$

Where:
*   $\kappa(A)$ is the **condition number** of the matrix.
*   $u_{\text{low}} \approx 10^{-7}$ is the machine epsilon of **FP32**.

If $\kappa(A) \gtrsim 10^7$, the low-precision solver cannot distinguish the solution signal from rounding noise, causing the process to stagnate or diverge.

---

### 4.4 GPU-Specific Performance Benefits
Implementing this foundation on a GPU provides two primary performance advantages:

*   **Memory Bandwidth Reduction:** Storing and reading the matrix $A$ (or its $LU$ factors) in **FP32** halves the required memory bandwidth compared to **FP64**. Since many solvers are bandwidth-bound, this leads to a near 2x speedup in data transfer.
*   **Hardware Throughput:** Modern GPUs (e.g., NVIDIA Ampere/Hopper) feature specialized units like **Tensor Cores** designed specifically for high-speed, low-precision matrix math. By offloading the $O(n^3)$ operations to these cores in FP32, the solver achieves significantly higher TFLOPS than is possible using the standard FP64 data path.

---

## 5. Code Structure

The following pseudo-code describes the integration of FP32 and FP64 operations within the GPU solver loop.

**Input:** Matrix $A$, Right-hand side $b$, Tolerance $\epsilon$  
**Output:** Refined solution $x$

```python
# --- Setup Phase (Low Precision) ---
A_f32 = convert_to_fp32(A)
b_f32 = convert_to_fp32(b)

# Perform LU Factorization in FP32 (O(n^3) - GPU Bottleneck)
L_f32, U_f32 = factorize_fp32(A_f32)

# Initial Solve
x_f64 = solve_fp32(L_f32, U_f32, b_f32) # Result promoted to FP64

# --- Refinement Phase (High Precision) ---
for k in range(max_iterations):
    # 1. Compute Residual in FP64 (Prevents cancellation)
    # r = b - Ax
    r_f64 = compute_residual_fp64(A, x_f64, b)
    
    # Check convergence
    if norm(r_f64) < epsilon:
        break
        
    # 2. Correction Solve in FP32 (Fast path)
    # Solve: A * delta_x = r
    r_f32 = convert_to_fp32(r_f64)
    delta_x_f32 = solve_fp32(L_f32, U_f32, r_f32)
    
    # 3. Update Solution in FP64
    # x = x + delta_x
    x_f64 = x_f64 + convert_to_fp64(delta_x_f32)

return x_f64
```
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

