#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <vector>
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

#include "gpu_image.h"

#include <tiny-cuda-nn/common_device.h>

using namespace tcnn;

__global__ void buffer_to_stbi_uc(uint32_t n_elements, network_precision_t* input, stbi_uc* output)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n_elements) {
		output[idx] = (stbi_uc)((float)input[idx] * 255.0f);
	}
}

void nrc::save_buffer_to_image(std::string filename, network_precision_t* data, uint32_t width, uint32_t height, uint32_t channels)
{
	uint32_t n_elements = width * height * channels;
	tcnn::GPUMemory<stbi_uc> img_gpu(n_elements);

	buffer_to_stbi_uc<<<n_blocks_linear(n_elements), n_threads_linear>>>(n_elements, data, img_gpu.data());
	std::vector<stbi_uc> img_cpu(n_elements);

	img_gpu.copy_to_host(img_cpu);

	stbi_write_png(filename.c_str(), width, height, channels, img_cpu.data(), 0);
}
