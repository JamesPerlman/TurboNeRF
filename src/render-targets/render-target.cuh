#pragma once

#include "../common.h"
#include "../utils/gpu-image.cuh"

#include <functional>
#include <string>
#include <vector>
#include <thread>

NRC_NAMESPACE_BEGIN

// For now, all Render Targets should be in RGBA8 format
class RenderTarget {
private:
    virtual void allocate(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) = 0;

    virtual void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) = 0;

    std::thread writeThread;

public:
    int width = 0;
    int height = 0;

    size_t n_pixels() const { return width * height; }

    RenderTarget() {};
    virtual ~RenderTarget() {
        if (writeThread.joinable()) {
            writeThread.join();
        }
    }

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
        if (writeThread.joinable()) {
            writeThread.join();
        }
        auto &imgWriteThread = writeThread;
        open_for_cuda_access([&](float* rgba) {
            save_buffer_to_image(stream, filename, rgba, width, height, 4, imgWriteThread);
        });
    };
};

NRC_NAMESPACE_END
