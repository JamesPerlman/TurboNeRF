#pragma once

#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"
#include "bounding-box.cuh"
#include "camera.cuh"
#include "cascaded-occupancy-grid.cuh"
#include "ray.h"
#include "workspace.cuh"


NRC_NAMESPACE_BEGIN

// NeRFWorkspace?
// TODO: Make this a derived struct from RenderingWorkspace
struct TrainingWorkspace: Workspace {
public:

	uint32_t batch_size;

	// arena properties
	BoundingBox* bounding_box;

	stbi_uc* image_data;

	float* random_float;
	uint32_t* img_index;
	uint32_t* pix_index; // index of randomly selected pixel in image

	uint32_t* ray_steps;
	uint32_t* ray_steps_cum;

	// ground-truth pixel color components
	float* pix_rgba;

	// accumulated ray sample color components
	float* ray_rgba;
	
	// ray origin components
	float* ray_origin;
	float* sample_origin;

	// ray direction components
	float* ray_dir;
	float* sample_dir;

	// ray inverse direction components
	float* ray_inv_dir;

	// ray t components
	float* sample_t0; // t_start
	float* sample_t1; // t_end
	float* sample_dt;

	// sample position components
	float* sample_pos;

	uint32_t n_occ_grid_elements;
	CascadedOccupancyGrid* occ_grid;
	uint8_t* occ_grid_bits;

	// GPUMemory managed properties

	tcnn::GPUMemory<Camera> cameras;

	// constructor
	TrainingWorkspace() {};

	// member functions
	void TrainingWorkspace::enlarge(
		const cudaStream_t& stream,
		const uint32_t& n_images,
		const uint32_t& n_pixels_per_image,
		const uint32_t& n_channels_per_image,
		const uint32_t& n_samples_per_batch,
		const uint32_t& n_occ_grid_levels,
		const uint32_t& n_occ_grid_cells_per_dimension
	) {
		free_allocations();
		
		batch_size = tcnn::next_multiple(n_samples_per_batch, tcnn::batch_size_granularity);
		uint32_t n_cameras = tcnn::next_multiple(n_images, tcnn::batch_size_granularity);
		uint32_t n_pixel_elements = tcnn::next_multiple(n_channels_per_image * n_pixels_per_image * n_images, tcnn::batch_size_granularity);

		n_occ_grid_elements = CascadedOccupancyGrid::get_n_total_elements(n_occ_grid_levels, n_occ_grid_cells_per_dimension);
		uint32_t n_grid_bitfield_bytes = tcnn::next_multiple(n_occ_grid_elements / 8, tcnn::batch_size_granularity);

		// need to upgrade to C++20 to use typename parameters in lambdas :(
		// auto alloc = []<typename T>(size_t size) { return allocate<T>(stream, size); };

		bounding_box 	= allocate<BoundingBox>(stream, 1);
		image_data 		= allocate<stbi_uc>(stream, n_pixel_elements);

		random_float 	= allocate<float>(stream, batch_size);

		img_index 		= allocate<uint32_t>(stream, batch_size);
		pix_index 		= allocate<uint32_t>(stream, batch_size);

		ray_steps 		= allocate<uint32_t>(stream, batch_size);
		ray_steps_cum 	= allocate<uint32_t>(stream, batch_size);

		pix_rgba 		= allocate<float>(stream, 4 * batch_size);
		ray_rgba 		= allocate<float>(stream, 4 * batch_size);

		ray_origin 		= allocate<float>(stream, 3 * batch_size);
		sample_origin 	= allocate<float>(stream, 3 * batch_size);
		ray_dir 		= allocate<float>(stream, 3 * batch_size);
		sample_dir 		= allocate<float>(stream, 3 * batch_size);
		ray_inv_dir 	= allocate<float>(stream, 3 * batch_size);

		sample_t0 		= allocate<float>(stream, batch_size);
		sample_t1 		= allocate<float>(stream, batch_size);
		sample_dt 		= allocate<float>(stream, batch_size);
		sample_pos 		= allocate<float>(stream, 3 * batch_size);

		occ_grid 		= allocate<CascadedOccupancyGrid>(stream, 1);
		occ_grid_bits 	= allocate<uint8_t>(stream, n_grid_bitfield_bytes);
	}


private:
	tcnn::GPUMemoryArena::Allocation arena_allocation;
};

NRC_NAMESPACE_END
