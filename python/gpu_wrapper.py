#!/usr/bin/env python3

import numpy as np
import ctypes

lib = ctypes.CDLL("./libmpsolver.so")

lib.gpuSolve.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

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

    lib.gpuSolve(
        A32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )

    return x32.astype(np.float64)


def gpu_refine(A, b, max_iter=10):
    n = A.shape[0]

    A64 = A.astype(np.float64).copy(order="C")
    b64 = b.astype(np.float64).copy(order="C")
    x64 = np.zeros_like(b64)

    lib.refineSolution(
        A64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        max_iter,
    )

    return x64