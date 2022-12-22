# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.24)

cmake_policy(SET CMP0104 NEW)

project (NeRFRenderCore LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

find_package(CUDA 12.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include sub-projects.
add_subdirectory(includes/eigen)
add_subdirectory(includes/json)

set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_BUILD_BENCHMARK OFF)
add_subdirectory(includes/tiny-cuda-nn)

include_directories(includes/stb)
link_directories(includes/stb)

# src project

set(HEADERS
	src/controllers/nerf-training-controller.h
	src/models/bounding-box.h
	src/models/camera.h
	src/models/dataset.h
	src/models/ray.h
	src/models/training-image.h
	src/models/json-bindings/eigen-json.h
	src/models/training-workspace.h
	src/common.h
	src/main.h
)

set(SOURCES
	src/controllers/nerf-training-controller.cu
	src/models/camera.cu
	src/models/dataset.cu
	src/models/training-image.cu
	src/models/training-workspace.cu
	src/utils/embedder.cu
	src/main.cu	
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# CUDA includes / linkage
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
	CUDA_STANDARD 17
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${CUDA_LIBRARIES}
	curand
	eigen
	tiny-cuda-nn
)
