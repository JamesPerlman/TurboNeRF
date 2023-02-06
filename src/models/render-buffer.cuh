#pragma once

#include <stdint.h>
#include <string>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct RenderBuffer {
public:
    uint32_t width;
    uint32_t height;
    uint32_t stride;

    // pointer to GPU memory to store the output data
    float* rgba;

    RenderBuffer(uint32_t width, uint32_t height, float* rgba)
        : width(width)
        , height(height)
        , stride(width * height)
        , rgba(rgba)
    {};

    void clear(const cudaStream_t& stream = 0);

    void save_image(const std::string& filename, const cudaStream_t& stream = 0);

    std::vector<float> fetch_image(const cudaStream_t& stream = 0);
};

NRC_NAMESPACE_END
