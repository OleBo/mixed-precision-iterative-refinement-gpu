cmake_minimum_required(VERSION 3.18)
project(mixed_precision_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Modern way to find CUDA toolkit libraries
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# just to be 100% sure the T4 code is being generated
set(CMAKE_CUDA_ARCHITECTURES 75) 

# 1. Build the library with all functional logic (Python cannot load a STATIC)
add_library(mixed_precision_lib SHARED
    src/solver.cu
    src/refinement.cu
)

# This tells the library to search for its dependencies in the CUDA folder at runtime
set_target_properties(mixed_precision_lib PROPERTIES
    INSTALL_RPATH "/usr/local/cuda/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 3. Link everything together
# We link the custom lib, then the CUDA toolkit libs
target_link_libraries(mixed_precision_lib 
    PRIVATE 
    CUDA::cublas 
    CUDA::cusolver
    CUDA::cudart
)

# ============================================================
# TESTING (GoogleTest)
# ============================================================
enable_testing()

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.12.1.zip
)

# Avoid Windows CRT issues (harmless elsewhere)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# -------------------------------
# Test executable
# -------------------------------
add_executable(mpsolver_tests
    tests/test_gpu_solve.cpp
    tests/test_refinement.cpp
)

# Important: tests are C++ (not CUDA), but link CUDA lib
target_link_libraries(mpsolver_tests
    PRIVATE
    mixed_precision_lib
    GTest::gtest_main
)

# Needed for CUDA runtime linking inside tests
target_link_libraries(mpsolver_tests
    PRIVATE
    CUDA::cudart
)

# Discover tests automatically
include(GoogleTest)
gtest_discover_tests(mpsolver_tests)