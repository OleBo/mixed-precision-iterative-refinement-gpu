// This is numerical software testing:
// - correctness (EXPECT_NEAR)
// - numerical validation (residuals)
// - robustness (random systems)
// - determinism
// - failure detection via info

#include <gtest/gtest.h>
#include "solver.h"

#include <vector>
#include <cmath>
#include <random>

// ------------------------------------------------------------
// Helper: compute residual ||Ax - b||_2
// ------------------------------------------------------------
float computeResidual(
    const std::vector<float>& A,
    const std::vector<float>& x,
    const std::vector<float>& b,
    int n)
{
    float residual = 0.0f;

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        float r = sum - b[i];
        residual += r * r;
    }

    return std::sqrt(residual);
}

// ------------------------------------------------------------
// Helper: deterministic diagonally dominant matrix generator
// ------------------------------------------------------------
void makeDeterministicSystem(
    std::vector<float>& A,
    std::vector<float>& b,
    int n,
    unsigned seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    A.resize(n * n);
    b.resize(n);

    for (int i = 0; i < n; i++) {
        float row_sum = 0.0f;

        for (int j = 0; j < n; j++) {
            float val = dist(gen);
            A[i * n + j] = val;
            row_sum += std::abs(val);
        }

        // enforce strict diagonal dominance → guaranteed nonsingular
        A[i * n + i] += row_sum + 1.0f;
    }

    for (int i = 0; i < n; i++) {
        b[i] = dist(gen);
    }
}

// ------------------------------------------------------------
// 1. Basic correctness (2x2 system)
// ------------------------------------------------------------
TEST(GpuSolveTest, Solves2x2System) {
    int n = 2;

    std::vector<float> A = {
        4, 1,
        2, 3
    };

    std::vector<float> b = {1, 1};
    std::vector<float> x(n, 0.0f);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_EQ(info, 0);
    EXPECT_NEAR(x[0], 0.2f, 1e-4f);
    EXPECT_NEAR(x[1], 0.2f, 1e-4f);
}

// ------------------------------------------------------------
// 2. Identity matrix
// ------------------------------------------------------------
TEST(GpuSolveTest, IdentityMatrix) {
    int n = 4;

    std::vector<float> A = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    std::vector<float> b = {3, -2, 5, 7};
    std::vector<float> x(n, 0.0f);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_EQ(info, 0);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], b[i], 1e-6f);
    }
}

// ------------------------------------------------------------
// 3. Zero RHS → zero solution
// ------------------------------------------------------------
TEST(GpuSolveTest, ZeroRHS) {
    int n = 3;

    std::vector<float> A = {
        3,1,2,
        1,4,0,
        2,0,5
    };

    std::vector<float> b(n, 0.0f);
    std::vector<float> x(n, 1.0f);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_EQ(info, 0);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], 0.0f, 1e-6f);
    }
}

// ------------------------------------------------------------
// 4. Residual check
// ------------------------------------------------------------
TEST(GpuSolveTest, ResidualIsSmall) {
    int n = 3;

    std::vector<float> A = {
        4, 1, 2,
        1, 3, 0,
        2, 0, 5
    };

    std::vector<float> b = {7, 4, 6};
    std::vector<float> x(n, 0.0f);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_EQ(info, 0);

    float res = computeResidual(A, x, b, n);

    EXPECT_LT(res, 1e-4f);
}

// ------------------------------------------------------------
// 5. Ill-conditioned system
// ------------------------------------------------------------
TEST(GpuSolveTest, IllConditionedSystem) {
    int n = 2;

    std::vector<float> A = {
        1.0f, 1.0f,
        1.0f, 1.0001f
    };

    std::vector<float> b = {2.0f, 2.0001f};
    std::vector<float> x(n, 0.0f);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_EQ(info, 0);

    float res = computeResidual(A, x, b, n);

    EXPECT_LT(res, 1e-2f);
}

// ------------------------------------------------------------
// 6. Deterministic random system
// ------------------------------------------------------------
TEST(GpuSolveTest, DeterministicRandomSystem) {
    int n = 5;

    std::vector<float> A, b;
    std::vector<float> x(n, 0.0f);

    makeDeterministicSystem(A, b, n, 42);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_EQ(info, 0);

    float res = computeResidual(A, x, b, n);

    EXPECT_LT(res, 1e-3f);
}

// ------------------------------------------------------------
// 7. Deterministic repeatability
// ------------------------------------------------------------
TEST(GpuSolveTest, DeterministicRepeatability) {
    int n = 5;

    std::vector<float> A, b;
    makeDeterministicSystem(A, b, n, 1337);

    std::vector<float> x1(n, 0.0f);
    std::vector<float> x2(n, 0.0f);

    int info1 = gpuSolve(A.data(), b.data(), x1.data(), n);
    int info2 = gpuSolve(A.data(), b.data(), x2.data(), n);

    EXPECT_EQ(info1, 0);
    EXPECT_EQ(info2, 0);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x1[i], x2[i], 1e-6f);
    }
}

// ------------------------------------------------------------
// 8. Singular matrix should fail
// ------------------------------------------------------------
TEST(GpuSolveTest, SingularMatrixFails) {
    int n = 2;

    std::vector<float> A = {
        1, 1,
        1, 1
    };

    std::vector<float> b = {2, 2};
    std::vector<float> x(n, 0.0f);

    int info = gpuSolve(A.data(), b.data(), x.data(), n);

    EXPECT_NE(info, 0);
}