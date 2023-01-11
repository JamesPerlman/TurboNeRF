#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <stbi/stb_image.h>

#include "../common.h"

#include <string>
#include <vector>

NRC_NAMESPACE_BEGIN

void save_buffer_to_image(
    const cudaStream_t& stream,
    const std::string& filename,
    const float* data,
    const uint32_t& width,
    const uint32_t& height,
    const uint32_t& channels,
    const uint32_t& stride = 0,
    const float& scale = 1.0f
);

NRC_NAMESPACE_END
