#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace mixed_precision {

void initializeCuda();
void shutdownCuda();

void gpuSolve(const float* A, const float* b, float* x, int n);
void refineSolution(const double* A, const double* b, double* x, int n, int maxIter);

} // namespace mixed_precision
