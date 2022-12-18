# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.24)

project (NeRFRenderCore LANGUAGES CXX CUDA)

cmake_policy(SET CMP0104 NEW)

# CUDA
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})


# Include sub-projects.
add_subdirectory(src)
add_subdirectory(includes)

message("SETTING")

# includes
target_link_libraries(NeRFRenderCore tiny-cuda-nn)
