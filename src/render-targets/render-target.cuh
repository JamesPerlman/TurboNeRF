#pragma once

#include "../common.h"
#include "../utils/gpu-image.cuh"

#include <functional>
#include <string>
#include <vector>

NRC_NAMESPACE_BEGIN

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

    virtual void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) = 0;

    virtual void open_for_cuda_access(std::function<void(float* rgba)> handle) = 0;

    void clear(const cudaStream_t& stream = 0) {
        open_for_cuda_access([&](float* rgba) {
            CUDA_CHECK_THROW(cudaMemsetAsync(rgba, 0, width * height * 4 * sizeof(float), stream));
        });
    };

    void save_image(const std::string& filename, const cudaStream_t& stream = 0) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        open_for_cuda_access([&](float* rgba) {
            save_buffer_to_image(stream, filename, rgba, width, height, 4, stride);
        });
    };

    std::vector<float> get_data(const cudaStream_t& stream = 0) {
        std::vector<float> data;
        open_for_cuda_access([&](float* rgba) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
            data = save_buffer_to_memory(stream, rgba, width, height, 4, stride);
        });
        return data;
    };
};

NRC_NAMESPACE_END
