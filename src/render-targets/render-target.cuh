#pragma once

#include "../common.h"
#include "../utils/gpu-image.cuh"

#include <functional>
#include <string>
#include <vector>

NRC_NAMESPACE_BEGIN

class RenderTarget {
private:
    virtual void allocate(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) = 0;

    virtual void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) = 0;

public:
    int width = 0;
    int height = 0;
    int stride = 0;

    RenderTarget() {};

    virtual void free(const cudaStream_t& stream = 0) = 0;

    virtual void open_for_cuda_access(std::function<void(float* rgba)> handle, const cudaStream_t& stream = 0) = 0;

    void set_size(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) {
        if (width == 0 || height == 0)
            return;
        
        if (this->width == 0 && this->height == 0) {
            allocate(width, height, stream);
        } else if (this->width != width || this->height != height) {
            resize(width, height, stream);
        }
    }

    void clear(const cudaStream_t& stream = 0) {
        open_for_cuda_access([&](float* rgba) {
            if (rgba == nullptr)
                return;
            
            CUDA_CHECK_THROW(cudaMemsetAsync(rgba, 0, width * height * 4 * sizeof(float), stream));
        });
    };

    void save_image(const std::string& filename, const cudaStream_t& stream = 0) {
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        open_for_cuda_access([&](float* rgba) {
            save_buffer_to_image(stream, filename, rgba, width, height, 4, stride);
        });
    };
};

NRC_NAMESPACE_END
