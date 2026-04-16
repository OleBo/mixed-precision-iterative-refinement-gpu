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
    src/utils.cu
)

# This tells the library to search for its dependencies in the CUDA folder at runtime
set_target_properties(mixed_precision_lib PROPERTIES
    INSTALL_RPATH "/usr/local/cuda/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Required for cross-file kernel calls and linking
set_target_properties(mixed_precision_lib PROPERTIES
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

# 3. Build the executable and link it to your lib
add_executable(mpsolver src/main.cu)
target_link_libraries(mpsolver PRIVATE mixed_precision_lib)