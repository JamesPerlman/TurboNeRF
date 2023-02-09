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

public:
    using RenderTarget::RenderTarget;
    
    void allocate(const cudaStream_t& stream = 0) override {
        CUDA_CHECK_THROW(cudaMallocAsync(&rgba, width * height * 4 * sizeof(float), stream));
    }

    void free(const cudaStream_t& stream = 0) override {
        CUDA_CHECK_THROW(cudaFreeAsync(rgba, stream));
    }

    void open_for_cuda_access(std::function<void(float* rgba)> handle) override {
        // this is a CUDA buffer with CUDA-allocated data, so there's nothing special we need to do here to prepare for readwrite access.
        handle(rgba);
    }

    void clear(const cudaStream_t& stream = 0) override {
        CUDA_CHECK_THROW(cudaMemsetAsync(rgba, 0, width * height * 4 * sizeof(float), stream));
    };

    void save_image(const std::string& filename, const cudaStream_t& stream = 0) const override {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        save_buffer_to_image(stream, filename, rgba, width, height, 4, stride);
    };

    std::vector<float> get_data(const cudaStream_t& stream = 0) const override {
        return save_buffer_to_memory(stream, rgba, width, height, 4, stride);
    };
};

NRC_NAMESPACE_END
