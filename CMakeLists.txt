cmake_minimum_required(VERSION 3.18)
project(mixed_precision_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Modern way to find CUDA toolkit libraries
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# 1. Build the library with all functional logic
add_library(mixed_precision_lib STATIC
    src/solver.cu
    src/refinement.cu
    src/utils.cu
)

# Required for cross-file kernel calls and linking
set_target_properties(mixed_precision_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 2. Build the executable
# IMPORTANT: src/main.cu (or whichever file has int main()) must be here
add_executable(mpsolver src/main.cu)

# 3. Link everything together
# We link the custom lib, then the CUDA toolkit libs
target_link_libraries(mpsolver 
    PRIVATE 
    mixed_precision_lib 
    CUDA::cublas 
    CUDA::cusolver
)
