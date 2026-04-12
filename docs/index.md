# Mixed-Precision Iterative Refinement Solver

A GPU-focused project showcasing:

- low-precision compute for speed
- high-precision residual correction for accuracy
- iterative refinement as a high-quality HPC design pattern

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
