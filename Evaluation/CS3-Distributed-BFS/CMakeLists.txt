cmake_minimum_required(VERSION 3.31)

project(SigmaTrace-Eval-3-BFS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif ()
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(NVSHMEM REQUIRED PATHS "${NVSHMEM_ROOT}" "${CUDAToolkit_LIBRARY_ROOT}/../../comm_libs/nvshmem")

add_executable(bfs_nvshmem_persist bfs_nvshmem_persist.cu)
set_target_properties(bfs_nvshmem_persist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bfs_nvshmem_persist PRIVATE CUDA::cudart)
target_link_libraries(bfs_nvshmem_persist PRIVATE nvshmem::nvshmem_host)
target_link_libraries(bfs_nvshmem_persist PRIVATE nvshmem::nvshmem_device)
