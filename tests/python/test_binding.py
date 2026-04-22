# Detects:
# - cuSOLVER failures (info != 0)
# - ABI mismatches
# - memory layout bugs (very common!)
# - silent GPU errors
# - nondeterminism
# Prevents:
# - “it works in C++ but crashes in Python”
# - silent NaNs / garbage results
# - wrong strides / non-contiguous arrays

import numpy as np
import pytest
from ctypes import cdll, c_float, c_int, POINTER

# ------------------------------------------------------------
# Load shared library
# ------------------------------------------------------------
lib = cdll.LoadLibrary("./build/libmixed_precision_lib.so")

# Define function signature
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

    assert A.dtype == np.float32
    assert b.dtype == np.float32
    assert A.flags["C_CONTIGUOUS"]
    assert b.flags["C_CONTIGUOUS"]

    x = np.zeros(n, dtype=np.float32)

    info = lib.gpuSolve(
        A.ctypes.data_as(POINTER(c_float)),
        b.ctypes.data_as(POINTER(c_float)),
        x.ctypes.data_as(POINTER(c_float)),
        n
    )

    if info != 0:
        raise RuntimeError(f"gpuSolve failed with info = {info}")

    return x


# ------------------------------------------------------------
# Helper: residual ||Ax - b||
# ------------------------------------------------------------
def residual(A, x, b):
    return np.linalg.norm(A @ x - b)


# ------------------------------------------------------------
# 1. Basic correctness
# ------------------------------------------------------------
def test_solve_2x2():
    A = np.array([[4, 1],
                  [2, 3]], dtype=np.float32)
    b = np.array([1, 1], dtype=np.float32)

    x = gpu_solve(A, b)

    assert np.allclose(x, [0.2, 0.2], atol=1e-4)


# ------------------------------------------------------------
# 2. Identity matrix
# ------------------------------------------------------------
def test_identity():
    A = np.eye(4, dtype=np.float32)
    b = np.array([3, -2, 5, 7], dtype=np.float32)

    x = gpu_solve(A, b)

    assert np.allclose(x, b, atol=1e-6)


# ------------------------------------------------------------
# 3. Residual check
# ------------------------------------------------------------
def test_residual_small():
    A = np.array([[4, 1, 2],
                  [1, 3, 0],
                  [2, 0, 5]], dtype=np.float32)
    b = np.array([7, 4, 6], dtype=np.float32)

    x = gpu_solve(A, b)

    assert residual(A, x, b) < 1e-4


# ------------------------------------------------------------
# 4. Ill-conditioned system
# ------------------------------------------------------------
def test_ill_conditioned():
    A = np.array([[1.0, 1.0],
                  [1.0, 1.0001]], dtype=np.float32)
    b = np.array([2.0, 2.0001], dtype=np.float32)

    x = gpu_solve(A, b)

    assert residual(A, x, b) < 1e-2


# ------------------------------------------------------------
# 5. Random system robustness
# ------------------------------------------------------------
def test_random_system():
    np.random.seed(42)
    n = 5

    A = np.random.uniform(-1, 1, (n, n)).astype(np.float32)

    # Make diagonally dominant
    for i in range(n):
        A[i, i] += np.sum(np.abs(A[i])) + 1.0

    b = np.random.uniform(-1, 1, n).astype(np.float32)

    x = gpu_solve(A, b)

    assert residual(A, x, b) < 1e-3


# ------------------------------------------------------------
# 6. Deterministic behavior
# ------------------------------------------------------------
def test_deterministic():
    A = np.array([[4, 1, 2],
                  [1, 3, 0],
                  [2, 0, 5]], dtype=np.float32)
    b = np.array([7, 4, 6], dtype=np.float32)

    x1 = gpu_solve(A, b)
    x2 = gpu_solve(A, b)

    assert np.allclose(x1, x2, atol=1e-6)


# ------------------------------------------------------------
# 7. Singular matrix → MUST fail
# ------------------------------------------------------------
def test_singular_matrix_fails():
    A = np.array([[1, 1],
                  [1, 1]], dtype=np.float32)
    b = np.array([2, 2], dtype=np.float32)

    with pytest.raises(RuntimeError):
        gpu_solve(A, b)


# ------------------------------------------------------------
# 8. ABI / memory layout test
# ------------------------------------------------------------
def test_non_contiguous_fails():
    A = np.array([[4, 1],
                  [2, 3]], dtype=np.float32).T  # non-contiguous
    b = np.array([1, 1], dtype=np.float32)

    with pytest.raises(AssertionError):
        gpu_solve(A, b)