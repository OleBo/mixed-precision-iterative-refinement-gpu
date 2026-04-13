#include "solver.h"

namespace mixed_precision {

// RAII helper to manage CUDA memory automatically within this file
template<typename T>
struct GpuPtr {
    T* ptr = nullptr;
    GpuPtr(size_t count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
    ~GpuPtr() { if (ptr) cudaFree(ptr); }
    operator T*() { return ptr; }
};

void initializeCuda() {
    CUDA_CHECK(cudaSetDevice(0));
}

void shutdownCuda() {
    cudaDeviceReset();
}

void gpuSolve(const float* d_A_in, const float* d_b_in, float* d_x_out, int n) {
    cusolverDnHandle_t handle;
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) return;

    // 1. Allocate internal buffers
    GpuPtr<float> d_A_copy(n * n);
    GpuPtr<int> d_pivot(n);
    GpuPtr<int> d_info(1);

    // 2. Setup Data: Sgetrf overwrites A; Sgetrs overwrites B (into x_out)
    CUDA_CHECK(cudaMemcpy(d_A_copy, d_A_in, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_out, d_b_in, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // 3. Workspace Query
    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle, n, n, d_A_copy, n, &lwork);
    GpuPtr<float> d_work(lwork);

    // 4. LU Factorization (A = PLU)
    cusolverDnSgetrf(handle, n, n, d_A_copy, n, d_work, d_pivot, d_info);

    // 5. Solve (Ax = b)
    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A_copy, n, d_pivot, d_x_out, n, d_info);

    cusolverDnDestroy(handle);
}

} // namespace mixed_precision
