#pragma once

#include <stdint.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct RenderBuffer {
public:
    uint32_t width;
    uint32_t height;

    // pointer to GPU memory to store the output data
    float* rgba;

    RenderBuffer(uint32_t width, uint32_t height, float* rgba)
        : width(width)
        , height(height)
        , rgba(rgba)
    {};
};

NRC_NAMESPACE_END
