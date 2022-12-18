# CMakeList.txt : CMake project for NeRFRenderCore, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable(NeRFRenderCore
	main.cu
	main.h
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET NeRFRenderCore PROPERTY CXX_STANDARD 20)
endif()

# CUDA includes / linkage

find_package(CUDA 12.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(NeRFRenderCore ${CUDA_LIBRARIES})
