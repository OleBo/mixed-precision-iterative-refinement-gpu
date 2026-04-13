#include "solver.h"
#include <iostream>
#include <vector>

int main() {
    int n = 100; // Matrix size
    int maxIter = 10;
    
    // Initialize host data
    std::vector<double> h_A(n * n, 1.0); // Simple dummy matrix
    std::vector<double> h_b(n, (double)n);
    std::vector<double> h_x(n, 0.0);

    // Make diagonal dominant so it's easy to solve
    for(int i = 0; i < n; ++i) h_A[i * n + i] = n * 2.0;

    // 1. Setup CUDA
    mixed_precision::initializeCuda();

    // 2. Run Refinement
    std::cout << "Starting mixed precision refinement..." << std::endl;
    mixed_precision::refineSolution(h_A.data(), h_b.data(), h_x.data(), n, maxIter);
    std::cout << "Done! First element of solution: " << h_x[0] << std::endl;

    // 3. Cleanup
    mixed_precision::shutdownCuda();

    return 0;
}
