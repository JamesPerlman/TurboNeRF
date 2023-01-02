#pragma once

#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"
#include "bounding-box.cuh"
#include "camera.h"
#include "cascaded-occupancy-grid.cuh"
#include "ray.h"


NRC_NAMESPACE_BEGIN

// NeRFWorkspace?
// TODO: Make this a derived struct from RenderingWorkspace
struct TrainingWorkspace {
public:

	uint32_t batch_size;

	// arena properties
	BoundingBox* bounding_box;

	stbi_uc* image_data;

	float* random_floats;
	uint32_t* img_index;
	uint32_t* pix_index; // index of randomly selected pixel in image

	uint32_t* ray_steps;
	uint32_t* ray_steps_cumulative;

	// ground-truth pixel color components
	float* pix_rgba;

	// accumulated ray sample color components
	float* ray_rgba;
	
	// ray origin components
	float* ray_origins;
	float* sample_origins;

	// ray direction components
	float* ray_dirs;
	float* sample_dirs;

	// ray inverse direction components
	float* ray_inv_dirs;

	// ray t components
	float* sample_t0; // t_start
	float* sample_t1; // t_end
	float* sample_dt;

	// sample position components
	float* sample_positions;

	uint32_t n_occupancy_grid_elements;
	CascadedOccupancyGrid* occupancy_grid;
	uint8_t* occupancy_grid_bitfield;

	// loss
	float* loss;

	// GPUMemory managed properties

	tcnn::GPUMemory<Camera> cameras;

	// constructor
	TrainingWorkspace() : arena_allocation() {};

	// member functions
	void enlarge(
		const cudaStream_t& stream,
		const uint32_t& n_images,
		const uint32_t& n_pixels_per_image,
		const uint32_t& n_channels_per_image,
		const uint32_t& n_samples_per_batch,
		const uint32_t& n_occupancy_grid_levels,
		const uint32_t& n_occupancy_grid_cells_per_dimension
	);

private:
	tcnn::GPUMemoryArena::Allocation arena_allocation;
};

NRC_NAMESPACE_END
