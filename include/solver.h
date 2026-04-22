#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------
// GPU LU solver (cuSOLVER-based)
// ------------------------------------------------------------
// Solves: A x = b
//
// A: row-major dense matrix (n x n)
// b: right-hand side vector (n)
// x: output solution vector (n)
//
// RETURNS:
//   0  -> success
//  >0  -> singular matrix (LU factorization failed)
//  <0  -> CUDA / cuSOLVER error
// ------------------------------------------------------------
int gpuSolve(float* A, float* b, float* x, int n);

// ------------------------------------------------------------
// Iterative refinement (mixed precision or double precision)
// ------------------------------------------------------------
// Improves an initial solution (internally or externally computed)
//
// A: row-major dense matrix (n x n)
// b: right-hand side vector (n)
// x: input initial guess + output refined solution
// max_iter: number of refinement iterations
// ------------------------------------------------------------
void refineSolution(double* A, double* b, double* x, int n, int max_iter);

#ifdef __cplusplus
}
#endif