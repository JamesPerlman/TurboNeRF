﻿# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.24)

cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0077 NEW)

project (TurboNeRF LANGUAGES CUDA CXX)

set(PYLIB_NAME PyTurboNeRF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# options
option(TN_BUILD_EXE "Build executable" ON)
option(TN_BUILD_PYD "Build Python module" OFF)

# we need at least one target to build
# if TN_BUILD_EXE is ON
if(TN_BUILD_EXE)
    set(BUILD_TARGET ${PROJECT_NAME})
elseif (TN_BUILD_PYD)
    set(BUILD_TARGET ${PYLIB_NAME})
endif()

if(NOT DEFINED BUILD_TARGET)
    message(STATUS "TN_BUILD_EXE is OFF and TN_BUILD_PYD is OFF. Nothing to build!")
    return()
endif()

# CUDA
if(DEFINED ENV{CUDA_ARCH_LIST})
    message(STATUS "Building for remote machine with CUDA architectures: $ENV{CUDA_ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDA_ARCH_LIST}")
else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    message(STATUS "Detected CUDA architectures: ${CUDA_ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()

set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# function to add CUDA archs as preprocessor definitions
function(add_cuda_archs_as_defines target)
    # Convert the list of architectures to a semicolon-separated string
    string(REPLACE ";" "," CUDA_ARCHS_STR "${CMAKE_CUDA_ARCHITECTURES}")

    # Pass the string as a preprocessor definition
    target_compile_definitions(${target} PRIVATE CUDA_ARCHS="${CUDA_ARCHS_STR}")
endfunction()

# Prefer CUDA 12, but use 11 if 12 not available
find_package(CUDA 12 QUIET)
if(NOT CUDA_FOUND)
    find_package(CUDA 11 QUIET)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA 12 (or 11) not found!")
    else()
        message(STATUS "Using CUDA 11 from ${CUDA_TOOLKIT_ROOT_DIR}")
    endif()    
else()
    message(STATUS "Using CUDA 12 from ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

include_directories(${CUDA_INCLUDE_DIRS})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# GLAD
add_subdirectory(includes/glad)
include_directories(includes/glad/include)

# tiny-cuda-nn
set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_BUILD_BENCHMARK OFF)
if(UNIX)
    # Enable generation of PIC code on Linux, otherwise
    # the tiny-cuda-nn library won't be able to link with
    # the Python module code
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
add_subdirectory(includes/tiny-cuda-nn)
if(UNIX)
    # And disable again
    set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()

# project includes
set(HEADERS
    # controllers
    src/controllers/nerf-rendering-controller.h
    src/controllers/nerf-training-controller.h

    # core
    src/core/adam-optimizer.cuh
    src/core/embedding.cuh
    src/core/lion-optimizer.cuh
    src/core/nerf-network.cuh
    src/core/occupancy-grid.cuh
    src/core/renderer.cuh
    src/core/trainer.cuh

    # effects
    src/effects/spatial-effect.cuh
    src/effects/spatial/repeater.cuh

    # math
    src/math/device-math.cuh
    src/math/hexagon-grid.cuh
    src/math/matrix4f.cuh
    src/math/transform4f.cuh
    src/math/tuple-math.cuh

    # models
    src/models/bounding-box.cuh
    src/models/camera.cuh
    src/models/dataset.h
    src/models/nerf.cuh
    src/models/nerf-proxy.cuh
    src/models/nerf-renderable.cuh
    src/models/ray.h
    src/models/ray-batch.cuh
    src/models/render-pattern.cuh
    src/models/render-modifiers.cuh
    src/models/render-request.cuh
    src/models/render-task.cuh
    src/models/training-image.cuh
    src/models/updatable-property.cuh

    # render-targets
    src/render-targets/cpu-render-buffer.cuh
    src/render-targets/cuda-render-buffer.cuh
    src/render-targets/opengl-render-surface.cuh
    src/render-targets/render-target.cuh

    # rendering
    src/rendering/ray-batch-coordinators/hexagonal-grid-ray-batch-coordinator.cuh
    src/rendering/ray-batch-coordinators/hexagonal-tile-ray-batch-coordinator.cuh
    src/rendering/ray-batch-coordinators/linear-buffer-ray-batch-coordinator.cuh
    src/rendering/ray-batch-coordinators/ray-batch-coordinator.cuh
    src/rendering/ray-batch-coordinators/rectangular-grid-ray-batch-coordinator.cuh
    src/rendering/render-task-factories/hexagonal-grid-render-task-factory.cuh
    src/rendering/render-task-factories/linear-chunk-render-task-factory.cuh
    src/rendering/render-task-factories/rectangular-grid-render-task-factory.cuh
    src/rendering/render-task-factories/render-task-factory.cuh

    # utils
    src/utils/bit-utils.cuh
    src/utils/camera-kernels.cuh
    src/utils/color-utils.cuh
    src/utils/common-network-kernels.cuh
    src/utils/gpu-image.cuh
    src/utils/nerf-constants.cuh
    src/utils/observable.cuh
    src/utils/occupancy-grid-kernels.cuh
    src/utils/parallel-utils.cuh
    src/utils/queues.h
    src/utils/raymarching-utils.cuh
    src/utils/render-flags.cuh
    src/utils/rendering-kernels.cuh
    src/utils/stream-compaction.cuh
    src/utils/training-batch-kernels.cuh
    src/utils/training-kernels-fused.cuh

    # services
    src/services/device-manager.cuh
    src/services/nerf-manager.cuh
    src/services/file-manager.cuh
    src/services/runtime-manager.cuh

    # workspaces
    src/workspaces/dataset-workspace.cuh
    src/workspaces/network-params-workspace.cuh
    src/workspaces/network-workspace.cuh
    src/workspaces/occupancy-grid-workspace.cuh
    src/workspaces/rendering-workspace.cuh
    src/workspaces/scene-workspace.cuh
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

    # effects
    src/effects/spatial/repeater.cu

    # models
    src/models/dataset.cu
    src/models/nerf.cu
    src/models/training-image.cu

    # rendering
    src/rendering/ray-batch-coordinators/hexagonal-grid-ray-batch-coordinator.cu
    src/rendering/ray-batch-coordinators/hexagonal-tile-ray-batch-coordinator.cu
    src/rendering/ray-batch-coordinators/linear-buffer-ray-batch-coordinator.cu
    src/rendering/ray-batch-coordinators/rectangular-grid-ray-batch-coordinator.cu
    src/rendering/render-task-factories/render-task-factory.cu

    # utils
    src/utils/bit-utils.cu
    src/utils/gpu-image.cu
    src/utils/occupancy-grid-kernels.cu
    src/utils/rendering-kernels.cu

    # common
    src/main.cu
)

set(TurboNeRF_LIBRARIES
    PRIVATE
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBRARIES}
    curand
    cublas
    glad
    tiny-cuda-nn
)

# Build Executable
if(TN_BUILD_EXE)
    message(STATUS "${PROJECT_NAME} Executable WILL be built!")
    add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
    
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )

    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${TurboNeRF_LIBRARIES}
    )

    # Set stricter compile flag (for Linux)
    if(NOT MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE -Xcompiler -Werror)
    endif()

    add_cuda_archs_as_defines(${PROJECT_NAME})
else()
    message(STATUS "${PROJECT_NAME} Executable will NOT be built!")
endif()


# Build Python Library
if (TN_BUILD_PYD)
    message(STATUS "${PYLIB_NAME} Python Library WILL be built!")
    # pybind11
    add_subdirectory(includes/pybind11)
    include_directories(${pybind11_INCLUDE_DIRS})

    set(PYBIND_INCLUDES
        src/api/pybind_cpp_utils.cuh
        src/api/pybind_cuda.cuh
        src/api/python_bindings.cu

        src/integrations/blender.cuh

        ${SOURCES}
        ${HEADERS}
    )

    pybind11_add_module(${PYLIB_NAME} ${PYBIND_INCLUDES})

    target_link_libraries(${PYLIB_NAME}
        PRIVATE
        ${TurboNeRF_LIBRARIES}
        pybind11::module
    )

    add_cuda_archs_as_defines(${PYLIB_NAME})
else()
    message(STATUS "${PYLIB_NAME} Python Library will NOT be built!")
endif()


# Copy DLLs to output directory
# This is super hacky, but I don't know how else to do it.
# I spent about 4 hours in a CMake rabbit hole trying to find a better way.
# - James

function(copy_cuda_dlls)
  # These are the DLL names
    set(CUDA_DLL_NAMES
        cublas64_12
        cudart64_12
        curand64_10
    )

    # Copy them to the output directory after building
    foreach(CUDA_DLL_NAME ${CUDA_DLL_NAMES})
        add_custom_command(TARGET ${BUILD_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/${CUDA_DLL_NAME}.dll"
            "$<TARGET_FILE_DIR:${BUILD_TARGET}>/"
        )
    endforeach()
endfunction()

if(MSVC)
  copy_cuda_dlls()
endif()
