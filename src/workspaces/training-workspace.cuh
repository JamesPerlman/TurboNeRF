#pragma once

#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"
#include "../core/occupancy-grid.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/ray.h"
#include "workspace.cuh"


TURBO_NAMESPACE_BEGIN

// NeRFWorkspace?
// TODO: Make this a derived struct from RenderingWorkspace
struct TrainingWorkspace: Workspace {

    using Workspace::Workspace;

	uint32_t batch_size;

	BoundingBox* bounding_box;
	OccupancyGrid* occupancy_grid;

	stbi_uc* image_data;

	// the data in this buffer maps pixel centers to their undistorted positions
	float* undistort_map;

	float* random_float;

	uint32_t* ray_step;
	uint32_t* ray_offset;

	// ground-truth pixel color components
	float* pix_rgba;

	// accumulated ray sample color components
	float* ray_rgba;
	
	// ray origin components
	float* ray_origin;
	float* sample_origin;

	// ray direction components
	float* ray_dir;

	// ray inverse direction components
	float* ray_inv_dir;

	// ray t components
	float* ray_t;
	float* ray_t_max;

	// ray alive is basically just a check for if the ray hits the bounding box
	bool* ray_alive;

	// ray_index is used for compaction while generating a training batch
	int* ray_index;

	// normalized network input
	tcnn::network_precision_t* network_concat;
	tcnn::network_precision_t* network_output;

	// sample buffers
	int* sample_index; // indices of samples (for compaction)
	bool* sample_visible; // is visible
	float* sample_pos;
    float* sample_dt;
	float* sample_dir;

	// GPUMemory managed properties
	tcnn::GPUMemory<Camera> cameras;

	// primitives
	size_t n_images;
	size_t n_pixels_per_image;
	uint32_t n_samples_per_batch;

	// member functions
	void enlarge(
		const cudaStream_t& stream,
		const size_t& n_images,
		const size_t& n_pixels_per_image,
		const uint32_t& n_samples_per_batch,
		const uint32_t& n_occ_grid_levels,
		const uint32_t& n_occ_grid_cells_per_dimension,
		const size_t& n_network_concat_elements,
		const size_t& n_network_output_elements
	) {
		free_allocations();

		this->n_images = n_images;
		this->n_pixels_per_image = n_pixels_per_image;
		this->n_samples_per_batch = n_samples_per_batch;
		
		batch_size = tcnn::next_multiple(n_samples_per_batch, tcnn::batch_size_granularity);
		size_t n_pixel_elements = 4 * n_pixels_per_image * n_images;

		// need to upgrade to C++20 to use typename parameters in lambdas :(
		// auto alloc = []<typename T>(size_t size) { return allocate<T>(stream, size); };

		occupancy_grid 	= allocate<OccupancyGrid>(stream, 1);
		bounding_box 	= allocate<BoundingBox>(stream, 1);
		image_data 		= allocate<stbi_uc>(stream, n_pixel_elements);

		undistort_map	= allocate<float>(stream, 2 * n_pixels_per_image);

		random_float 	= allocate<float>(stream, 4 * batch_size);

		pix_rgba		= allocate<float>(stream, 4 * batch_size);
		ray_rgba 		= allocate<float>(stream, 4 * batch_size);

		ray_step		= allocate<uint32_t>(stream, batch_size);
		ray_offset		= allocate<uint32_t>(stream, batch_size);
		ray_origin 		= allocate<float>(stream, 3 * batch_size);
		ray_dir			= allocate<float>(stream, 3 * batch_size);
		ray_inv_dir 	= allocate<float>(stream, 3 * batch_size);
		ray_t 			= allocate<float>(stream, batch_size);
		ray_t_max		= allocate<float>(stream, batch_size);
		ray_alive 		= allocate<bool>(stream, batch_size);
		ray_index 		= allocate<int>(stream, batch_size);

		sample_index	= allocate<int>(stream, batch_size);
		sample_pos		= allocate<float>(stream, 3 * batch_size);
		sample_dt		= allocate<float>(stream, batch_size);
		sample_dir		= allocate<float>(stream, 3 * batch_size);
		sample_visible	= allocate<bool>(stream, batch_size);

		network_concat	= allocate<tcnn::network_precision_t>(stream, n_network_concat_elements * batch_size);
		network_output	= allocate<tcnn::network_precision_t>(stream, n_network_output_elements * batch_size);
	}
};

TURBO_NAMESPACE_END
