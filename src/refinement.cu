#include "solver.h"

#include <vector>
#include <cmath>
#include <iostream>

// ------------------------------------------------------------
// Helper: compute residual r = b - A x  (double precision)
// ------------------------------------------------------------
static void computeResidual(
    const double* A,
    const double* x,
    const double* b,
    double* r,
    int n)
{
    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }

        r[i] = b[i] - sum;
    }
}

// ------------------------------------------------------------
// Helper: norm ||r||_2
// ------------------------------------------------------------
static double norm2(const std::vector<double>& r)
{
    double s = 0.0;
    for (double v : r) {
        s += v * v;
    }
    return std::sqrt(s);
}

// ------------------------------------------------------------
// Mixed-precision iterative refinement
// ------------------------------------------------------------
extern "C"
void refineSolution(
    double* A,     // double precision matrix
    double* b,     // double precision RHS
    double* x,     // solution (updated in-place)
    int n,
    int max_iter)
{
    // --------------------------------------------------------
    // Initial guess: solve in single precision
    // --------------------------------------------------------
    std::vector<float> A_f(n * n);
    std::vector<float> b_f(n);
    std::vector<float> x_f(n);

    for (int i = 0; i < n * n; i++)
        A_f[i] = static_cast<float>(A[i]);

    for (int i = 0; i < n; i++)
        b_f[i] = static_cast<float>(b[i]);

    int info = gpuSolve(A_f.data(), b_f.data(), x_f.data(), n);

    if (info != 0) {
        std::cerr << "Initial solve failed, info = " << info << std::endl;
        return;
    }

    for (int i = 0; i < n; i++)
        x[i] = static_cast<double>(x_f[i]);

    // --------------------------------------------------------
    // Iterative refinement loop
    // --------------------------------------------------------
    std::vector<double> r(n);        // residual (double)
    std::vector<float>  r_f(n);      // residual (float)
    std::vector<float>  delta_f(n);  // correction (float)

    for (int iter = 0; iter < max_iter; iter++) {

        // Compute residual r = b - A x
        computeResidual(A, x, b, r.data(), n);

        double res_norm = norm2(r);

        // Early exit if already good
        if (res_norm < 1e-12) {
            break;
        }

        // Convert residual to single precision
        for (int i = 0; i < n; i++)
            r_f[i] = static_cast<float>(r[i]);

        // Solve A * delta = r  (reuse single precision solver)
        info = gpuSolve(A_f.data(), r_f.data(), delta_f.data(), n);

        if (info != 0) {
            std::cerr << "Refinement step failed at iter "
                      << iter << ", info = " << info << std::endl;
            return;
        }

        // Update solution: x += delta
        for (int i = 0; i < n; i++)
            x[i] += static_cast<double>(delta_f[i]);
    }
}