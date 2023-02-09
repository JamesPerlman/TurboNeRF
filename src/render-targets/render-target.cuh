#pragma once

#include "../common.h"

#include <functional>
#include <string>
#include <vector>

class RenderTarget {
public:
    int width;
    int height;
    int stride;

    RenderTarget(const uint32_t& width, const uint32_t& height)
        : width(width)
        , height(height)
        , stride(width * height)
    {};

    virtual void allocate(const cudaStream_t& stream = 0) = 0;

    virtual void free(const cudaStream_t& stream = 0) = 0;

    virtual void open_for_cuda_access(std::function<void(float* rgba)> handle) = 0;

    virtual void clear(const cudaStream_t& stream = 0) = 0;

    virtual void save_image(const std::string& filename, const cudaStream_t& stream = 0) const = 0;

    virtual std::vector<float> get_data(const cudaStream_t& stream = 0) const = 0;
};
