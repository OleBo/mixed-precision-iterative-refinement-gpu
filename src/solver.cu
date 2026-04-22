// - performs LU factorization with cuSOLVER
// - solves Ax=b
// - copies d_info back to host
// - returns info so your tests can assert on it

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <iostream>

// Optional: simple CUDA error macro (lightweight but useful)
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << std::endl;                           \
            goto cleanup;                                        \
        }                                                     \
    } while (0)

#define CUSOLVER_CHECK(call)                                  \
    do {                                                      \
        cusolverStatus_t status = (call);                     \
        if (status != CUSOLVER_STATUS_SUCCESS) {              \
            std::cerr << "cuSOLVER error: " << status         \
                      << std::endl;                           \
            goto cleanup;                                        \
        }                                                     \
    } while (0)

extern "C"
int gpuSolve(float* A, float* b, float* x, int n)
{
    float *d_A = nullptr;
    float *d_b = nullptr;
    int *d_pivots = nullptr;
    int *d_info = nullptr;

    int info = 0;
    int work_size = 0;
    float* d_work = nullptr;

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pivots, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // Step 1: LU factorization A = P * L * U
    // ------------------------------------------------------------
    CUSOLVER_CHECK(
        cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n, &work_size)
    );

    CUDA_CHECK(cudaMalloc((void**)&d_work, work_size * sizeof(float)));

    CUSOLVER_CHECK(
        cusolverDnSgetrf(
            handle,
            n,
            n,
            d_A,
            n,
            d_work,
            d_pivots,
            d_info
        )
    );

    // Copy info back after factorization
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info != 0) {
        // LU failed (e.g. singular matrix)
        std::cerr << "LU factorization failed, info = " << info << std::endl;
        goto cleanup;
    }

    // ------------------------------------------------------------
    // Step 2: Solve Ax = b using LU
    // ------------------------------------------------------------
    CUSOLVER_CHECK(
        cusolverDnSgetrs(
            handle,
            CUBLAS_OP_T,
            n,
            1,
            d_A,
            n,
            d_pivots,
            d_b,
            n,
            d_info
        )
    );

    // Copy info again (solve phase)
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info != 0) {
        std::cerr << "Solve failed, info = " << info << std::endl;
        goto cleanup;
    }

    // Copy result x = solution (stored in d_b)
    CUDA_CHECK(cudaMemcpy(x, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));

cleanup:
    if (d_work)    cudaFree(d_work);
    if (d_A)       cudaFree(d_A);
    if (d_b)       cudaFree(d_b);
    if (d_pivots)  cudaFree(d_pivots);
    if (d_info)    cudaFree(d_info);

    cusolverDnDestroy(handle);

    return info;
}