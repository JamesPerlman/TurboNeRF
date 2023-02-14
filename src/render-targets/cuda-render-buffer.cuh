#pragma once

#include <stdint.h>
#include <string>

#include "../common.h"
#include "../utils/gpu-image.cuh"
#include "render-target.cuh"

NRC_NAMESPACE_BEGIN

class CUDARenderBuffer: public RenderTarget {
private:
    // pointer to GPU memory to store the output data
    float* rgba = nullptr;
    
    void allocate(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        this->width = width;
        this->height = height;

        CUDA_CHECK_THROW(cudaMallocAsync(&rgba, width * height * 4 * sizeof(float), stream));
    }

    void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        free(stream);
        allocate(width, height, stream);
    }

public:
    using RenderTarget::RenderTarget;
    
    void free(const cudaStream_t& stream = 0) override {
        if (width == 0 || height == 0)
            return;
        
        CUDA_CHECK_THROW(cudaFreeAsync(rgba, stream));
    }

    void open_for_cuda_access(std::function<void(float* rgba)> handle, const cudaStream_t& stream = 0) override {
        // this is a CUDA buffer with CUDA-allocated data, so there's nothing special we need to do here to prepare for readwrite access.
        handle(rgba);
    }
};

NRC_NAMESPACE_END
