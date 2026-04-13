#include "solver.h"
#include <cuda_runtime.h>
#include <iostream>

namespace mixed_precision {

/**
 * Enhanced error printing for .cu files.
 * Provides a standardized way to log failures without stopping execution
 * unless you decide to throw an exception.
 */
void printCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA DEVICE ERROR] %s\n", msg);
        std::fprintf(stderr, "  Error String: %s\n", cudaGetErrorString(err));
        std::fprintf(stderr, "  Error Name:   %s\n", cudaGetErrorName(err));
    }
}

} // namespace mixed_precision
