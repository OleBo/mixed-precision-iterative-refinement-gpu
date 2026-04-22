#!/usr/bin/env python3

import numpy as np
import pytest
from ctypes import cdll, c_float, c_double, c_int, POINTER

# ------------------------------------------------------------
# Load shared library
# ------------------------------------------------------------
lib = cdll.LoadLibrary("./build/libmixed_precision_lib.so")

# Define gpuSolve signature
lib.gpuSolve.argtypes = [
    POINTER(c_float),  # A
    POINTER(c_float),  # b
    POINTER(c_float),  # x
    c_int              # n
]
lib.gpuSolve.restype = c_int


# ------------------------------------------------------------
# Helper: call solver with safety checks
# ------------------------------------------------------------
def gpu_solve(A: np.ndarray, b: np.ndarray):
    n = A.shape[0]

    # Ensure data is float precision (float32) and contiguous
    #assert A.dtype == np.float32
    #assert b.dtype == np.float32
    #assert A.flags["C_CONTIGUOUS"]
    #assert b.flags["C_CONTIGUOUS"]
    A = A.astype(np.float32).copy(order="C")
    b = b.astype(np.float32).copy(order="C")
   

    x = np.zeros_like(b)

    info = lib.gpuSolve(
        A.ctypes.data_as(POINTER(c_float)),
        b.ctypes.data_as(POINTER(c_float)),
        x.ctypes.data_as(POINTER(c_float)),
        n
    )

    if info != 0:
        raise RuntimeError(f"gpuSolve failed with info = {info}")

    return x


# # Define refineSolution signature
lib.refineSolution.argtypes = [
    POINTER(c_double), # h_A
    POINTER(c_double), # h_b
    POINTER(c_double), # h_x
    c_int,              # n
    c_int               # maxIter
]


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
        A64.ctypes.data_as(POINTER(c_double)),
        b64.ctypes.data_as(POINTER(c_double)),
        x64.ctypes.data_as(POINTER(c_double)),
        n,
        max_iter,
    )

    return x64