FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c")


find_package(CUDA QUIET REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(SYSTEM ${PROJECT_SOURCE_DIR} ${MPI_INCLUDE_PATH})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lcufftw -lcufft -lculibos)
set(CUDA_ARCH_LIST "Auto" CACHE STRING "List of CUDA architectures (e.g. 52, 61, 75)")
include(FindCUDA/select_compute_arch)

CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

cuda_add_executable(TiltRec-cuda main-cuda.cpp)
cuda_add_executable(TiltRecZ-cuda main-cudaz.cpp)
add_executable(TiltRec-mpi main-mpi.cpp)
add_executable(TiltRecZ-mpi main-mpiz.cpp)


target_link_libraries(TiltRec-cuda ${MPI_LIBRARIES} mrcmx filter recon_y opts)
target_link_libraries(TiltRecZ-cuda ${MPI_LIBRARIES} mrcmx filter recon_z opts)
target_link_libraries(TiltRec-mpi ${MPI_LIBRARIES} mrcmx filter)
target_link_libraries(TiltRecZ-mpi ${MPI_LIBRARIES} mrcmx filter)


