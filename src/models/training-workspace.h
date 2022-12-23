#pragma once

#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"

#include "camera.h"
#include "ray.h"


NRC_NAMESPACE_BEGIN

// NeRFWorkspace?
struct TrainingWorkspace {
public:

	uint32_t batch_size;

	// arena properties
	stbi_uc* image_data;
	tcnn::network_precision_t* density_input;
	tcnn::network_precision_t* density_output;
	tcnn::network_precision_t* color_output;
	float* random_floats;
	uint32_t* image_indices;
	uint32_t* pixel_indices;
	float* rgb_batch;
	float* ray_dir_batch;

	// GPUMemory managed properties

	tcnn::GPUMemory<Camera> cameras;

	// constructor
	TrainingWorkspace() : arena_allocation() {};

	// member functions
	void enlarge(cudaStream_t stream, uint32_t n_images, uint32_t n_pixels_per_image, uint32_t n_channels_per_image, uint32_t training_batch_size);

private:
	tcnn::GPUMemoryArena::Allocation arena_allocation;
};

NRC_NAMESPACE_END
