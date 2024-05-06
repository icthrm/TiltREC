FILE(GLOB RECON_Z_SRCS "*.cu" "*.cpp" "*.c")

MESSAGE(STATUS "building recon_z")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/local/cuda-11.7/targets/x86_64-linux/include)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

set(CUDA_ARCH_LIST "Auto" CACHE STRING "List of CUDA architectures (e.g. 52, 61, 75)")
include(FindCUDA/select_compute_arch)
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})


cuda_add_library (recon_z STATIC ${RECON_Z_SRCS})

target_link_libraries(recon_z filter mrcmx)

set_target_properties(recon_z PROPERTIES OUTPUT_NAME "recon_z")
set_target_properties(recon_z PROPERTIES CLEAN_DIRECT_OUTPUT 1)
