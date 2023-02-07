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

# glfw
add_subdirectory(includes/glfw)
include_directories(includes/glfw/include)

# pybind11
add_subdirectory(includes/pybind11)
include_directories(pybind11_INCLUDE_DIRS)

# Include sub-projects.

set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_BUILD_BENCHMARK OFF)
add_subdirectory(includes/tiny-cuda-nn)

# src project

set(HEADERS
	# controllers
	src/controllers/nerf-rendering-controller.h
	src/controllers/nerf-training-controller.h

	# core
	src/core/occupancy-grid.cuh
	src/core/nerf-network.cuh
	src/core/renderer.cuh
	src/core/trainer.cuh

	# integrations
	src/integrations/blender.cuh

	# models
	src/models/bounding-box.cuh
	src/models/camera.cuh
	src/models/dataset.h
	src/models/nerf.cuh
	src/models/nerf-proxy.cuh
	src/models/ray.h
	src/models/render-buffer.cuh
	src/models/render-request.cuh
	src/models/training-image.cuh

	# utils
	src/utils/bit-utils.cuh
	src/utils/camera-kernels.cuh
	src/utils/color-utils.cuh
	src/utils/common-network-kernels.cuh
	src/utils/coordinate-transformations.cuh
	src/utils/gpu-image.cuh
	src/utils/linalg/transform4f.cuh
	src/utils/linalg.cuh
	src/utils/nerf-constants.cuh
	src/utils/occupancy-grid-kernels.cuh
	src/utils/parallel-utils.cuh
	src/utils/rendering-kernels.cuh
	src/utils/stream-compaction.cuh
	src/utils/training-batch-kernels.cuh
	src/utils/training-network-kernels.cuh

	# services
	src/services/device-manager.cuh
	src/services/nerf-manager.cuh

	# workspaces
	src/workspaces/network-params-workspace.cuh
	src/workspaces/network-workspace.cuh
	src/workspaces/occupancy-grid-workspace.cuh
	src/workspaces/rendering-workspace.cuh
	src/workspaces/training-workspace.cuh
	src/workspaces/workspace.cuh

	# common
	src/common.h
	src/main.h
)

set(SOURCES
	# controllers
	src/controllers/nerf-rendering-controller.cu
	src/controllers/nerf-training-controller.cu

	# core
	src/core/nerf-network.cu
	src/core/renderer.cu
	src/core/trainer.cu

	# models
	src/models/dataset.cu
	src/models/render-buffer.cu
	src/models/training-image.cu

	# utils
	src/utils/gpu-image.cu
	src/utils/occupancy-grid-kernels.cu
	src/utils/rendering-kernels.cu
	src/utils/stream-compaction.cu

	# common
	src/main.cu	
)

# Build results
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
pybind11_add_module(PyNeRFRenderCore src/api/python_bindings.cu ${SOURCES} ${HEADERS})

# Set stricter compile flag
target_compile_options(${PROJECT_NAME} PRIVATE -Xcompiler)

# CUDA includes / linkage
set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_SEPARABLE_COMPILATION ON
	INTERPROCEDURAL_OPTIMIZATION TRUE
)

set(NeRFRenderCore_LIBRARIES
	PRIVATE
	${CUDA_LIBRARIES}
	curand
	cublas
	glfw
	tiny-cuda-nn
)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
	${NeRFRenderCore_LIBRARIES}
	pybind11::module
)

target_link_libraries(PyNeRFRenderCore
	PRIVATE
	${NeRFRenderCore_LIBRARIES}
	pybind11::module
)
