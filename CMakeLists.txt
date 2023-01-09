# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.25.1)


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

set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_BUILD_BENCHMARK OFF)
add_subdirectory(includes/tiny-cuda-nn)


# src project

set(HEADERS
	# controllers
	src/controllers/nerf-rendering-controller.h
	src/controllers/nerf-training-controller.h

	# models
	src/models/bounding-box.cuh
	src/models/camera.cuh
	src/models/cascaded-occupancy-grid.cuh
	src/models/cascaded-occupancy-grid-workspace.cuh
	src/models/dataset.h
	src/models/nerf-network-workspace.cuh
	src/models/nerf-network.h
	src/models/nerf.cuh
	src/models/ray.h
	src/models/rays.h
	src/models/render-buffer.cuh
	src/models/render-request.cuh
	src/models/rendering-workspace.cuh
	src/models/training-image.cuh
	src/models/training-workspace.cuh
	src/models/workspace.cuh

	# utils
	src/utils/gpu-image.cuh
	src/utils/bit-utils.cuh
	src/utils/coordinate-transformations.cuh
	src/utils/linalg.cuh
	src/utils/parallel-utils.cuh
	src/utils/training-batch-kernels.cuh
	src/utils/training-network-kernels.cuh
	src/utils/volumetric-rendering.cuh

	# services
	src/services/nerf-manager.cuh

	# common
	src/common.h
	src/main.h
)

set(SOURCES
	src/controllers/nerf-rendering-controller.cu
	src/controllers/nerf-training-controller.cu
	src/models/dataset.cu
	src/models/nerf-network.cu
	src/models/training-image.cu
	src/main.cu	
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# CUDA includes / linkage
set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
	${CUDA_LIBRARIES}
	curand
	cublas
	tiny-cuda-nn
)
