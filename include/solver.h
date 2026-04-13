#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

namespace mixed_precision {

// --- Lifecycle Management ---
void initializeCuda();
void shutdownCuda();

// --- Solvers ---
// Note: gpuSolve now uses float* for device pointers to match your refinement loop
void gpuSolve(const float* d_A, const float* d_b, float* d_x, int n);
void refineSolution(const double* A, const double* b, double* x, int n, int maxIter);

// --- Error Handling Utilities ---
void printCudaError(cudaError_t err, const char* msg);

/**
 * Universal Macro for CUDA API calls.
 * Wraps the call and prints file/line info on failure.
 */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error [" << __FILE__ << ":" << __LINE__     \
                      << "]: " << cudaGetErrorString(err) << std::endl;     \
        }                                                                  \
    } while (0)

/**
 * Macro for cuBLAS API calls (since they return cublasStatus_t, not cudaError_t).
 */
#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = call;                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " (Status " << status << ")" << std::endl;        \
        }                                                                  \
    } while (0)

} // namespace mixed_precision
