cmake_minimum_required(VERSION 3.18)
project(mixed_precision_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(mixed_precision_solver
    src/solver.cu
    src/refinement.cu
    src/utils.cu
)

set_target_properties(mixed_precision_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(mpsolver src/solver.cu)
target_link_libraries(mpsolver PRIVATE mixed_precision_solver cublas cusolver)
