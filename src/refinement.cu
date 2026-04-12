#include "solver.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace mixed_precision {

// This file contains the mixed precision refinement driver.

void refineSolution(const double* A, const double* b, double* x, int n, int maxIter) {
    // Allocate GPU memory
    double *d_A, *d_b, *d_x, *d_r;
    float *d_A_float, *d_b_float, *d_r_float, *d_dx;

    cudaError_t err;
    err = cudaMalloc(&d_A, n * n * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMalloc(&d_b, n * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_b: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc(&d_x, n * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_x: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        return;
    }
    err = cudaMalloc(&d_r, n * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_r: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        return;
    }
    err = cudaMalloc(&d_A_float, n * n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_A_float: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        return;
    }
    err = cudaMalloc(&d_b_float, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_b_float: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_A_float);
        return;
    }
    err = cudaMalloc(&d_r_float, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_r_float: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_A_float);
        cudaFree(d_b_float);
        return;
    }
    err = cudaMalloc(&d_dx, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate d_dx: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_A_float);
        cudaFree(d_b_float);
        cudaFree(d_r_float);
        return;
    }

    // Copy A and b to GPU
    err = cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy A to device: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    err = cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy b to device: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }

    // Convert A and b to float on host
    float* A_float = new float[n * n];
    float* b_float = new float[n];
    for (int i = 0; i < n * n; i++) A_float[i] = (float)A[i];
    for (int i = 0; i < n; i++) b_float[i] = (float)b[i];

    err = cudaMemcpy(d_A_float, A_float, n * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy A_float to device: " << cudaGetErrorString(err) << std::endl;
        delete[] A_float;
        delete[] b_float;
        goto cleanup;
    }
    err = cudaMemcpy(d_b_float, b_float, n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy b_float to device: " << cudaGetErrorString(err) << std::endl;
        delete[] A_float;
        delete[] b_float;
        goto cleanup;
    }

    // Initial solve in single precision
    gpuSolve(d_A_float, d_b_float, d_dx, n);

    // Copy dx to x
    float* dx = new float[n];
    err = cudaMemcpy(dx, d_dx, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy dx from device: " << cudaGetErrorString(err) << std::endl;
        delete[] A_float;
        delete[] b_float;
        delete[] dx;
        goto cleanup;
    }
    for (int i = 0; i < n; i++) x[i] = (double)dx[i];

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        delete[] A_float;
        delete[] b_float;
        delete[] dx;
        goto cleanup;
    }

    // Iterative refinement
    double alpha = -1.0;
    double beta = 1.0;
    for (int iter = 0; iter < maxIter; iter++) {
        // Compute residual r = b - A*x
        err = cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy x to device: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        err = cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy b to d_r: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        status = cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_A, n, d_x, 1, &beta, d_r, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS dgemv failed" << std::endl;
            break;
        }

        // Compute norm of residual
        double r_norm;
        status = cublasDnrm2(handle, n, d_r, 1, &r_norm);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS Dnrm2 failed" << std::endl;
            break;
        }
        if (r_norm < 1e-12) break; // convergence

        // Convert residual to float
        double* r_host = new double[n];
        err = cudaMemcpy(r_host, d_r, n * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy r to host: " << cudaGetErrorString(err) << std::endl;
            delete[] r_host;
            break;
        }
        for (int i = 0; i < n; i++) b_float[i] = (float)r_host[i]; // reuse b_float
        err = cudaMemcpy(d_r_float, b_float, n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy r_float to device: " << cudaGetErrorString(err) << std::endl;
            delete[] r_host;
            break;
        }

        // Solve correction
        gpuSolve(d_A_float, d_r_float, d_dx, n);

        // Update x
        err = cudaMemcpy(dx, d_dx, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy dx from device: " << cudaGetErrorString(err) << std::endl;
            delete[] r_host;
            break;
        }
        for (int i = 0; i < n; i++) x[i] += (double)dx[i];

        delete[] r_host;
    }

    cublasDestroy(handle);
    delete[] A_float;
    delete[] b_float;
    delete[] dx;

cleanup:
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_A_float);
    cudaFree(d_b_float);
    cudaFree(d_r_float);
    cudaFree(d_dx);
}

} // namespace mixed_precision
