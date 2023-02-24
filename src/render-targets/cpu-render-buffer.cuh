#pragma once

#include <stdint.h>
#include <string>

#include "../common.h"
#include "render-target.cuh"

NRC_NAMESPACE_BEGIN

class CPURenderBuffer: public RenderTarget {
private:
    float* rgba_cpu = nullptr;
    float* rgba_gpu = nullptr;
    bool _dirty = false;

    void allocate(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        this->width = width;
        this->height = height;

        CUDA_CHECK_THROW(cudaMallocHost(&rgba_cpu, width * height * 4 * sizeof(float)));
        CUDA_CHECK_THROW(cudaMalloc(&rgba_gpu, width * height * 4 * sizeof(float)));
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
        
        CUDA_CHECK_THROW(cudaFreeHost(rgba_cpu));
        CUDA_CHECK_THROW(cudaFree(rgba_gpu));
    }

    void open_for_cuda_access(std::function<void(float* rgba)> handle, const cudaStream_t& stream = 0) override {
        // allow writing to device memory
        handle(rgba_gpu);
        _dirty = true;
    }

    void synchronize() {
        // copy to CPU memory if needed
        if (_dirty) {
            CUDA_CHECK_THROW(cudaMemcpy(rgba_cpu, rgba_gpu, width * height * 4 * sizeof(float), cudaMemcpyDeviceToHost));
            _dirty = false;
        }
    }

    const float* get_rgba() {
        return rgba_cpu;
    }
};

NRC_NAMESPACE_END
