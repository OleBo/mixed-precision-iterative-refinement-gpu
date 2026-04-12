#include "solver.h"
#include <cuda_runtime.h>
#include <iostream>

namespace mixed_precision {

void printCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    }
}

} // namespace mixed_precision
