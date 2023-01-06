#pragma once
#ifndef NRC_GPU_IMAGE_H
#define NRC_GPU_IMAGE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

#include "parallel-utils.cuh"

#include <string>
#include <vector>



NRC_NAMESPACE_BEGIN

__global__ void buffer_to_stbi_uc(uint32_t n_elements, const float* __restrict__ input, stbi_uc* __restrict__ output, float scale)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n_elements) {
		output[idx] = (stbi_uc)((float)input[idx] / scale * 255.0f);
	}
}

__global__ void decontigify_kernel(uint32_t n_elements, const stbi_uc* __restrict__ input, stbi_uc* __restrict__ output, uint32_t stride, uint32_t n_channels)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n_elements) {
        output[(idx / n_channels) + (idx % n_channels) * stride] = input[idx];
    }
}

template <typename T>
void save_buffer_to_image(cudaStream_t stream, std::string filename, T* data, uint32_t width, uint32_t height, uint32_t channels, uint32_t stride = 0, float scale = 1.0f)
{
	uint32_t n_elements = width * height * channels;


	tcnn::GPUMemory<stbi_uc> img_gpu(n_elements);
    tcnn::GPUMemory<float> data_float(n_elements);

    std::vector<stbi_uc> img_cpu(n_elements);

    if (std::is_same<T, stbi_uc>::value) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(img_cpu.data(), data, n_elements * sizeof(stbi_uc), cudaMemcpyDeviceToHost, stream));
    } else {

        copy_and_cast(stream, n_elements, data_float.data(), data);

        buffer_to_stbi_uc<<<tcnn::n_blocks_linear(n_elements), tcnn::n_threads_linear, 0, stream>>>(n_elements, data_float.data(), img_gpu.data(), scale);

        if (stride > 0) {
            tcnn::GPUMemory<stbi_uc> img_gpu_decontig(n_elements);
            decontigify_kernel<<<tcnn::n_blocks_linear(n_elements), tcnn::n_threads_linear, 0, stream>>>(n_elements, img_gpu.data(), img_gpu_decontig.data(), channels, stride);
            img_gpu_decontig.copy_to_host(img_cpu);
        } else {
            img_gpu.copy_to_host(img_cpu);
        }
    }

	stbi_write_png(filename.c_str(), width, height, channels, img_cpu.data(), width * sizeof(stbi_uc) * channels);
}

NRC_NAMESPACE_END

#endif