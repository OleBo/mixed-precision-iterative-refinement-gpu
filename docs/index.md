# Mixed-Precision Iterative Refinement Solver

A GPU-focused project showcasing:

- low-precision compute for speed
- high-precision residual correction for accuracy
- iterative refinement as a high-quality HPC design pattern

I've implemented the mixed precision refinement driver in `refinement.cu`. The implementation performs iterative refinement using a combination of single-precision (float) and double-precision (double) arithmetic:

- **Initial solve**: Uses single-precision LU factorization via cuSOLVER to get an approximate solution
- **Iterative refinement loop**: 
  - Computes the residual in double precision using a custom CUDA kernel with shared memory tiling and coalesced memory access for optimal performance
  - Solves the correction equation in single precision
  - Updates the solution in double precision
  - Checks for convergence based on residual norm

The core idea of iterative refinement is:

- compute residual: $r_k = b - A x_k$
- solve correction: $A \, \delta x_k = r_k$
- update solution: $x_{k+1} = x_k + \delta x_k$

In the current code, the residual is computed in FP64, while the correction solve is performed in lower precision and then added back to the double-precision solution.

The custom matrix-vector multiplication kernel uses:
- Shared memory tiling to cache vector elements and reduce global memory accesses
- Coalesced memory access patterns for efficient data loading from global memory
- Block size of 32 threads for balanced occupancy and performance

I've also implemented the `gpuSolve` function in solver.cu using cuSOLVER's LU factorization and solve routines for single-precision linear systems.

The code includes proper CUDA memory management, error checking, and cleanup. Note that for production use, you might want to optimize by pre-factorizing the single-precision matrix once instead of factorizing it in each `gpuSolve` call.

## GitHub Repository

The source code for this project is available on GitHub: [mixed-precision-iterative-refinement-gpu][def_repo]

## What you will find here

- CPU reference baseline in `python/experiments.py`
- GPU solver skeleton in `src/`
- mixed precision iterative refinement architecture
- data and visualization placeholders in `results/`

## Pages

This documentation is published via GitHub Pages from `docs/`.

## Next steps

1. Build the CUDA project using `CMake`
2. Run the Python baseline experiments
3. Extend `src/refinement.cu` with iterative refinement kernels


[def_repo]: https://github.com/OleBo/mixed-precision-iterative-refinement-gpu