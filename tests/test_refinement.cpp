// This suite will catch:
// - incorrect LU reuse
// - wrong precision transitions
// - broken residual computation
// - non-converging refinement
// - memory / indexing bugs
// - numerical instability

#include <gtest/gtest.h>
#include "solver.h"

#include <vector>
#include <cmath>
#include <random>

// ------------------------------------------------------------
// Helper: compute residual ||Ax - b||_2
// ------------------------------------------------------------
double computeResidual(
    const std::vector<double>& A,
    const std::vector<double>& x,
    const std::vector<double>& b,
    int n)
{
    double residual = 0.0;

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        double r = sum - b[i];
        residual += r * r;
    }

    return std::sqrt(residual);
}

// ------------------------------------------------------------
// Helper: deterministic diagonally dominant system
// ------------------------------------------------------------
void makeDeterministicSystem(
    std::vector<double>& A,
    std::vector<double>& b,
    int n,
    unsigned seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    A.resize(n * n);
    b.resize(n);

    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;

        for (int j = 0; j < n; j++) {
            double val = dist(gen);
            A[i * n + j] = val;
            row_sum += std::abs(val);
        }

        // enforce strict diagonal dominance
        A[i * n + i] += row_sum + 1.0;
    }

    for (int i = 0; i < n; i++) {
        b[i] = dist(gen);
    }
}

// ------------------------------------------------------------
// 1. Basic correctness
// ------------------------------------------------------------
TEST(RefinementTest, Solves2x2SystemAccurately) {
    int n = 2;

    std::vector<double> A = {
        4, 1,
        2, 3
    };

    std::vector<double> b = {1, 1};
    std::vector<double> x(n, 0.0);

    refineSolution(A.data(), b.data(), x.data(), n, 10);

    EXPECT_NEAR(x[0], 0.2, 1e-10);
    EXPECT_NEAR(x[1], 0.2, 1e-10);
}

// ------------------------------------------------------------
// 2. Identity matrix
// ------------------------------------------------------------
TEST(RefinementTest, IdentityMatrix) {
    int n = 4;

    std::vector<double> A = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    std::vector<double> b = {3.0, -2.0, 5.0, 7.0};
    std::vector<double> x(n, 0.0);

    refineSolution(A.data(), b.data(), x.data(), n, 5);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], b[i], 1e-12);
    }
}

// ------------------------------------------------------------
// 3. Residual must be small
// ------------------------------------------------------------
TEST(RefinementTest, ResidualIsSmall) {
    int n = 3;

    std::vector<double> A = {
        4, 1, 2,
        1, 3, 0,
        2, 0, 5
    };

    std::vector<double> b = {7, 4, 6};
    std::vector<double> x(n, 0.0);

    refineSolution(A.data(), b.data(), x.data(), n, 10);

    double res = computeResidual(A, x, b, n);

    EXPECT_LT(res, 1e-10);
}

// ------------------------------------------------------------
// 4. Ill-conditioned system
// ------------------------------------------------------------
TEST(RefinementTest, IllConditionedSystem) {
    int n = 2;

    std::vector<double> A = {
        1.0, 1.0,
        1.0, 1.0001
    };

    std::vector<double> b = {2.0, 2.0001};
    std::vector<double> x(n, 0.0);

    refineSolution(A.data(), b.data(), x.data(), n, 20);

    double res = computeResidual(A, x, b, n);

    EXPECT_LT(res, 1e-8);
}

// ------------------------------------------------------------
// 5. Convergence improves solution
// ------------------------------------------------------------
TEST(RefinementTest, ConvergenceImprovesSolution) {
    int n = 3;

    std::vector<double> A = {
        4, 1, 2,
        1, 3, 0,
        2, 0, 5
    };

    std::vector<double> b = {7, 4, 6};

    std::vector<double> x1(n, 0.0);
    std::vector<double> x2(n, 0.0);

    refineSolution(A.data(), b.data(), x1.data(), n, 1);
    refineSolution(A.data(), b.data(), x2.data(), n, 10);

    double res1 = computeResidual(A, x1, b, n);
    double res2 = computeResidual(A, x2, b, n);

    EXPECT_LE(res2, res1);
}

// ------------------------------------------------------------
// 6. Deterministic random system
// ------------------------------------------------------------
TEST(RefinementTest, DeterministicRandomSystem) {
    int n = 5;

    std::vector<double> A, b;
    std::vector<double> x(n, 0.0);

    makeDeterministicSystem(A, b, n, 42);

    refineSolution(A.data(), b.data(), x.data(), n, 10);

    double res = computeResidual(A, x, b, n);

    EXPECT_LT(res, 1e-8);
}

// ------------------------------------------------------------
// 7. Deterministic repeatability
// ------------------------------------------------------------
TEST(RefinementTest, DeterministicRepeatability) {
    int n = 5;

    std::vector<double> A, b;
    makeDeterministicSystem(A, b, n, 1337);

    std::vector<double> x1(n, 0.0);
    std::vector<double> x2(n, 0.0);

    refineSolution(A.data(), b.data(), x1.data(), n, 10);
    refineSolution(A.data(), b.data(), x2.data(), n, 10);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x1[i], x2[i], 1e-12);
    }
}

// ------------------------------------------------------------
// 8. Zero RHS
// ------------------------------------------------------------
TEST(RefinementTest, ZeroRHS) {
    int n = 3;

    std::vector<double> A = {
        3,1,2,
        1,4,0,
        2,0,5
    };

    std::vector<double> b(n, 0.0);
    std::vector<double> x(n, 1.0);

    refineSolution(A.data(), b.data(), x.data(), n, 10);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], 0.0, 1e-12);
    }
}