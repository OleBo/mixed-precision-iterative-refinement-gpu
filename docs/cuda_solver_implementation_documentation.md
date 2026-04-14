# CUDA C++ Implementation — Mixed Precision Solver

## 1. Overview

This document explains the **CUDA C++ implementation** of the mixed-precision linear solver with a focus on:

- GPU-specific design decisions
- Memory management and allocation
- Parallelization strategy
- Use of templates (where applicable)
- Interaction with cuBLAS / cuSOLVER

The implementation consists of:
- `solver.cu` — GPU-based linear solve (FP32)
- `refinement.cu` — mixed-precision iterative refinement
- `solver.h` — interface and utilities

---

## 2. High-Level Execution Flow

### GPU Solve (FP32)

1. Copy matrix \(A\) and vector \(b\) to device memory
2. Factorize \(A\) (typically LU)
3. Solve \(Ax = b\)
4. Copy solution back

### Refinement Loop

1. Compute residual in FP64 (CPU or GPU)
2. Cast residual to FP32
3. Solve correction system on GPU
4. Update solution in FP64

---

## 3. Memory Management in CUDA

### 3.1 Device Memory Allocation

Typical pattern:

```cpp
float* d_A;
cudaMalloc((void**)&d_A, n * n * sizeof(float));
```

Key points:
- Memory resides on GPU (global memory)
- Must be explicitly allocated and freed
- Expensive compared to CPU allocation

---

### 3.2 Host–Device Transfers

```cpp
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
```

Performance implications:
- PCIe transfer is a bottleneck
- Should be minimized

---

### 3.3 RAII and Templates (Best Practice)

Although not fully shown in the header, a modern CUDA design often uses templates:

```cpp
template<typename T>
struct DeviceArray {
    T* data;
    size_t size;

    DeviceArray(size_t n) : size(n) {
        cudaMalloc(&data, n * sizeof(T));
    }

    ~DeviceArray() {
        cudaFree(data);
    }
};
```

Advantages:
- Type safety (float vs double)
- Automatic cleanup (RAII)
- Reusable abstraction

---

## 4. Use of Templates in the Solver

### 4.1 Motivation

Templates allow writing **precision-independent code**:

- `float` (FP32)
- `double` (FP64)

Example:

```cpp
template<typename T>
void copyToDevice(T* d, const T* h, int n);
```

---

### 4.2 Mixed Precision Pattern

Your implementation uses:
- FP32 for GPU solves
- FP64 for refinement

Templates could generalize this to:

```cpp
template<typename Low, typename High>
void refine(...);
```

---

### 4.3 Benefits

- Avoid code duplication
- Enable compile-time optimization
- Allow easy experimentation with precisions

---

## 5. Parallelization Strategy

### 5.1 CUDA Execution Model

- Threads are grouped into **blocks**
- Blocks form a **grid**

Each thread executes the same kernel on different data.

---

### 5.2 Data Parallelism in Linear Algebra

For matrix operations:

- Each thread computes one element or partial sum
- Example (matrix-vector multiply):

```cpp
__global__ void matvec(float* A, float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0;
        for (int j = 0; j < n; ++j)
            sum += A[i*n + j] * x[j];
        y[i] = sum;
    }
}
```

---

### 5.3 cuBLAS / cuSOLVER Usage

Instead of writing kernels manually, the code uses:

- cuBLAS → BLAS operations (GEMM, GEMV)
- cuSOLVER → factorizations (LU, QR)

Advantages:
- Highly optimized
- Uses advanced tiling and caching internally

---

## 6. Memory Hierarchy and Tiling

### 6.1 GPU Memory Types

| Memory Type | Speed | Scope |
|------------|------|------|
| Registers | Fastest | Per-thread |
| Shared Memory | Very fast | Per-block |
| Global Memory | Slow | All threads |

---

### 6.2 Memory Tiling Concept

Tiling improves performance by:
- Loading small blocks of data into shared memory
- Reusing them across threads

---

### 6.3 Example: Tiled Matrix Multiply

```cpp
__shared__ float tileA[32][32];
__shared__ float tileB[32][32];
```

Steps:
1. Load tiles from global memory
2. Synchronize threads
3. Compute partial products
4. Repeat for next tile

---

### 6.4 In Your Solver

Even if not explicitly written:
- cuBLAS/cuSOLVER internally use tiling
- LU factorization is blocked for cache efficiency

---

## 7. Numerical Kernels

### 7.1 LU Factorization (GPU)

The solver likely performs:

\[
A = LU
\]

Then solves:

\[
Ly = b, \quad Ux = y
\]

---

### 7.2 Stability Considerations

- Pivoting is essential
- FP32 introduces rounding errors

---

## 8. Mixed Precision Refinement (Implementation View)

Key steps in code:

1. Compute residual (double precision)
2. Convert residual to float
3. Call `gpuSolve`
4. Update solution (double precision)

---

## 9. Performance Considerations

### 9.1 Bottlenecks

- Memory transfers (CPU ↔ GPU)
- Kernel launch overhead

---

### 9.2 Optimization Strategies

- Keep data on GPU as long as possible
- Use batched operations
- Avoid repeated allocations

---

## 10. Error Handling

Macros:

```cpp
CUDA_CHECK(...)
CUBLAS_CHECK(...)
```

Provide:
- Debugging support
- Runtime diagnostics

---

## 11. Key GPU Insight

> Performance comes from parallelism + memory locality, not just FLOPs.

Your solver leverages:
- Massive parallelism (GPU cores)
- Optimized libraries (cuBLAS/cuSOLVER)
- Mixed precision to reduce cost

---

## 12. Summary

This CUDA implementation demonstrates:

- Efficient GPU-based linear algebra
- Proper separation of precision roles
- Use of industrial-grade libraries
- A scalable design for HPC workloads

---

**Status**: CUDA Implementation Documentation — COMPLETE

