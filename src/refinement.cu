#include "solver.h"
#include <vector>

#define TILE_SIZE 32

namespace mixed_precision {

// --- Kernels ---
__global__ void matvec_kernel(const double* A, const double* x, double* y, int n) {
    __shared__ double x_shared[TILE_SIZE];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    
    double sum = 0.0;
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int col_start = t * TILE_SIZE;
        if (threadIdx.x < TILE_SIZE && col_start + threadIdx.x < n)
            x_shared[threadIdx.x] = x[col_start + threadIdx.x];
        else
            x_shared[threadIdx.x] = 0.0;
        __syncthreads();
        
        for (int j = 0; j < TILE_SIZE && col_start + j < n; ++j)
            sum += A[row * n + col_start + j] * x_shared[j];
        __syncthreads();
    }
    y[row] = sum;
}

__global__ void subtract_kernel(const double* b, const double* Ax, double* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - Ax[i];
}

__global__ void update_kernel(double* x, const float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += (double)dx[i];
}

__global__ void cast_to_float(const double* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

// --- Main Refinement Driver ---
void refineSolution(const double* h_A, const double* h_b, double* h_x, int n, int maxIter) {
    // 1. Memory Management (using the header's CUDA_CHECK)
    double *d_A, *d_b, *d_x, *d_r, *d_temp;
    float *d_A_f, *d_b_f, *d_r_f, *d_dx;

    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A_f, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r_f, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, n * sizeof(float)));

    // 2. Initial Setup: Transfer and Pre-cast A to float
    std::vector<float> h_A_f(n * n);
    for (int i = 0; i < n * n; ++i) h_A_f[i] = (float)h_A[i];

    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_f, h_A_f.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));

    // 3. cuBLAS for Norms
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    dim3 block(TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE);

    // 4. Initial Solution Solve
    cast_to_float<<<grid, block>>>(d_b, d_b_f, n);
    gpuSolve(d_A_f, d_b_f, d_dx, n);
    update_kernel<<<grid, block>>>(d_x, d_dx, n);

    // 5. Refinement Loop
    for (int iter = 0; iter < maxIter; iter++) {
        // Residual: r = b - Ax (Double Precision)
        matvec_kernel<<<grid, block>>>(d_A, d_x, d_temp, n);
        subtract_kernel<<<grid, block>>>(d_b, d_temp, d_r, n);

        // Convergence Check
        double r_norm;
        CUBLAS_CHECK(cublasDnrm2(handle, n, d_r, 1, &r_norm));
        if (r_norm < 1e-12) break;

        // Solve for Correction: A * dx = r (Single Precision)
        cast_to_float<<<grid, block>>>(d_r, d_r_f, n);
        gpuSolve(d_A_f, d_r_f, d_dx, n);

        // Update Solution: x = x + dx (Double Precision)
        update_kernel<<<grid, block>>>(d_x, d_dx, n);
    }

    // 6. Return Result
    CUDA_CHECK(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup (Omitted manual free for brevity; use GpuPtr struct for better RAII)
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_r);
    cudaFree(d_temp); cudaFree(d_A_f); cudaFree(d_b_f);
    cudaFree(d_r_f); cudaFree(d_dx);
}

} // namespace mixed_precision
