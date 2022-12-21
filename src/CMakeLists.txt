# CMakeList.txt : CMake project for NeRFRenderCore, include source and define
# project specific logic here.
#

# Add source to this project's executable.

set(HEADERS
	controllers/nerf-training-controller.h
	models/bounding-box.h
	models/camera.h
	models/dataset.h
	models/ray.h
	models/training-image.h
	models/json-bindings/eigen-json.h
	models/training-workspace.h
	common.h
	main.h
)

set(SOURCES
	controllers/nerf-training-controller.cu
	models/camera.cu
	models/dataset.cu
	models/training-image.cu
	models/training-workspace.cu
	utils/embedder.cu
	main.cu	
)


add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# CUDA includes / linkage
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
	CUDA_STANDARD 17
)

find_package(CUDA 12.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} curand)