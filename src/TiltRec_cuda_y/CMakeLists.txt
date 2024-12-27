FILE(GLOB RECON_Y_SRCS "*.cu" "*.c" "*.cuh")

MESSAGE(STATUS "building recon_y")

find_package(CUDA QUIET REQUIRED)

include_directories(SYSTEM ${PROJECT_SOURCE_DIR})
include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lcufftw -lcufft -lculibos  --extended-lambda)

set(CUDA_ARCH_LIST "Auto" CACHE STRING "List of CUDA architectures (e.g. 52, 61, 75)")
include(FindCUDA/select_compute_arch)
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})


cuda_add_library(recon_y STATIC ${RECON_Y_SRCS})
target_link_libraries(recon_y filter mrcmx)

set_target_properties(recon_y PROPERTIES OUTPUT_NAME "recon_y")
set_target_properties(recon_y PROPERTIES CLEAN_DIRECT_OUTPUT 1)
