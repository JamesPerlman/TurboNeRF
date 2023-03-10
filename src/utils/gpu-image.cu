
#include "gpu-image.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>
#include <vector>

#include "parallel-utils.cuh"

TURBO_NAMESPACE_BEGIN

__global__ void buffer_to_stbi_uc(
    const uint32_t n_elements,
    const float* __restrict__ input,
    stbi_uc* __restrict__ output,
    const float scale
) {
	const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n_elements) {
		output[idx] = (stbi_uc)((float)input[idx] / scale * 255.0f);
	}
}

template <typename T>
__global__ void join_channels_kernel(
    const uint32_t n_pixels,
    const int n_channels,
    const T* __restrict__ input,
    T* __restrict__ output
) {
    const uint32_t pix_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (pix_idx >= n_pixels) return;

    const uint32_t j_idx = n_channels * pix_idx;
    
    uint32_t c_idx = pix_idx;
    
    for (int i = 0; i < n_channels; ++i) {
        output[j_idx + i] = input[c_idx];
        c_idx += n_pixels;
    }
}

void save_buffer_to_image(
    const cudaStream_t& stream,
    const std::string& filename,
    const float* data,
    const uint32_t& width,
    const uint32_t& height,
    const uint32_t& channels,
    const uint32_t& stride,
    const float& scale
) {
    const uint32_t n_pixels = width * height;
	const uint32_t n_elements = n_pixels * channels;

	tcnn::GPUMemory<stbi_uc> img_gpu(n_elements);
    tcnn::GPUMemory<float> data_float(n_elements);

    std::vector<stbi_uc> img_cpu(n_elements);

    if (std::is_same<float, stbi_uc>::value) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(img_cpu.data(), data, n_elements * sizeof(stbi_uc), cudaMemcpyDeviceToHost, stream));
    } else {

        copy_and_cast(stream, n_elements, data_float.data(), data);

        buffer_to_stbi_uc<<<tcnn::n_blocks_linear(n_elements), tcnn::n_threads_linear, 0, stream>>>(n_elements, data_float.data(), img_gpu.data(), scale);

        if (stride > 0) {
            tcnn::GPUMemory<stbi_uc> img_gpu_decontig(n_elements);
            join_channels_kernel<<<tcnn::n_blocks_linear(n_pixels), tcnn::n_threads_linear, 0, stream>>>(n_pixels, channels, img_gpu.data(), img_gpu_decontig.data());
            img_gpu_decontig.copy_to_host(img_cpu);
        } else {
            img_gpu.copy_to_host(img_cpu);
        }
    }

	stbi_write_png(filename.c_str(), width, height, channels, img_cpu.data(), width * sizeof(stbi_uc) * channels);
}

std::vector<float> save_buffer_to_memory(
    const cudaStream_t& stream,
    const float* data,
    const uint32_t& width,
    const uint32_t& height,
    const uint32_t& channels,
    const uint32_t& stride,
    const float& scale
) {
    const uint32_t n_pixels = width * height;
    const uint32_t n_elements = n_pixels * channels;


    std::vector<float> img_cpu(n_elements);

    if (stride > 0) {
        tcnn::GPUMemory<float> img_gpu_decontig(n_elements);
        join_channels_kernel<<<tcnn::n_blocks_linear(n_pixels), tcnn::n_threads_linear, 0, stream>>>(n_pixels, channels, data, img_gpu_decontig.data());
        img_gpu_decontig.copy_to_host(img_cpu);
    } else {
        cudaMemcpy(img_cpu.data(), data, n_elements * sizeof(float), cudaMemcpyDeviceToHost);
    }

    return img_cpu;
}

void join_channels(
    const cudaStream_t& stream,
    const uint32_t& width,
    const uint32_t& height,
    const uint32_t& channels,
    const float* data,
    float* output
) {
    const uint32_t n_pixels = width * height;

    join_channels_kernel<<<tcnn::n_blocks_linear(n_pixels), tcnn::n_threads_linear, 0, stream>>>(n_pixels, channels, data, output);
}

TURBO_NAMESPACE_END
