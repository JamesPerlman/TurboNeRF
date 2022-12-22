#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"

#include "ray.h"


NRC_NAMESPACE_BEGIN

// NeRFWorkspace?
struct TrainingWorkspace {
public:
	TrainingWorkspace() : arena_allocation() {};
	Ray* rays;
	tcnn::network_precision_t* density_input;
	tcnn::network_precision_t* density_output;
	tcnn::network_precision_t* color_output;
	float* random_floats;
	uint32_t* image_indices;
	uint32_t* pixel_indices;
	float* rgb_batch;
	Ray* ray_batch;
	
	uint32_t batch_size;
	
	void enlarge(cudaStream_t stream, uint32_t n_pixels, uint32_t n_images, uint32_t training_batch_size);

private:
	tcnn::GPUMemoryArena::Allocation arena_allocation;
};

NRC_NAMESPACE_END
