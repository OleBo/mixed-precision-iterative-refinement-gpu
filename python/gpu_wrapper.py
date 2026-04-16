#!/usr/bin/env python3

import numpy as np
import ctypes
import os

# 1. Manually add the CUDA library path to the environment
os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/lib64'

# 2. Use the absolute path to your .so file 
# Colab paths can be tricky with relative './build'
so_path = "/content/drive/MyDrive/mixed-precision-iterative-refinement-gpu/build/libmixed_precision_lib.so"

# Load the shared library
#lib = ctypes.CDLL("./build/libmixed_precision_lib.so")
lib = ctypes.CDLL(so_path)

# Define the argument types for gpuSolve
# Assuming: void gpuSolve(float* A, float* b, float* x, int n)
lib.gpuSolve.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

# Define argument types
# void refineSolution(const double* h_A, const double* h_b, double* h_x, int n, int maxIter)
lib.refineSolution.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
]


def gpu_solve_fp32(A, b):
    n = A.shape[0]

    A32 = A.astype(np.float32).copy(order="C")
    b32 = b.astype(np.float32).copy(order="C")
    x32 = np.zeros_like(b32)

    # Call the CUDA function
    lib.gpuSolve(
        # Get pointers to the numpy arrays
        A32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )

    return x32.astype(np.float64)


def gpu_refine(A, b, max_iter=10):
    n = A.shape[0]

    # Ensure data is double precision (float64) and contiguous
    # A = np.ascontiguousarray(A, dtype=np.float64)
    # b = np.ascontiguousarray(b, dtype=np.float64)
    A64 = A.astype(np.float64).copy(order="C")
    b64 = b.astype(np.float64).copy(order="C")
    x64 = np.zeros_like(b64)

    # Call C++
    lib.refineSolution(
        A64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        max_iter,
    )

    return x64