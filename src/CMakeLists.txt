# CMakeList.txt : CMake project for NeRFRenderCore, include source and define
# project specific logic here.
#

# Add source to this project's executable.

set(HEADERS
	common.h
	main.h
)

set(SOURCES
	main.cu	
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

include_directories(
	data-loader
	json-bindings
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# CUDA includes / linkage

find_package(CUDA 12.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
