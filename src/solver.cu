#include "solver.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

namespace mixed_precision {

void initializeCuda() {
    int device = 0;
    cudaSetDevice(device);
}

void shutdownCuda() {
    cudaDeviceReset();
}

void gpuSolve(const float* A, const float* b, float* x, int n) {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // Allocate device memory
    float *d_A, *d_b;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    // Copy data
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // LU factorization
    int *d_pivot, *d_info;
    cudaMalloc(&d_pivot, n * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    int lwork;
    cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, n, &lwork);
    float *d_work;
    cudaMalloc(&d_work, lwork * sizeof(float));

    cusolverDnSgetrf(cusolverH, n, n, d_A, n, d_work, d_pivot, d_info);

    // Solve
    cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, 1, d_A, n, d_pivot, d_b, n, d_info);

    // Copy result
    cudaMemcpy(x, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_pivot);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
}

} // namespace mixed_precision
