#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "bounding-box.cuh"
#include "camera.cuh"
#include "cascaded-occupancy-grid.cuh"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct RenderingWorkspace: Workspace {

	uint32_t batch_size;
	
	// misc
	Camera* camera;
	BoundingBox* bounding_box;
	CascadedOccupancyGrid* occupancy_grid;

	// compaction
	int c_block_size;
	int c_n_blocks;
	int* c_block_counts;
	int* c_block_offsets;
	int* compact_idx;

	// rays
	bool* ray_alive;
	bool* ray_active[2];
	float* ray_origin[2];
	float* ray_dir[2];
	float* ray_idir[2];
	float* ray_t[2];
	float* ray_sigma[2]; // accumulated sigma

	// 2D ray index (x + y * width)
	uint32_t* ray_idx[2]; 
	
	// samples
	float* sample_pos;
	float* sample_dt;

	// network buffers
	tcnn::network_precision_t* network_sigma;
	tcnn::network_precision_t* network_color;

	// output buffers
	float* pixel_buffer;

	// samples
	void enlarge(
		const cudaStream_t& stream,
		const uint32_t& output_width,
		const uint32_t& output_height,
		const uint32_t& n_elements_per_batch,
		const uint32_t& n_network_sigma_elements,
		const uint32_t& n_network_color_elements,
		const uint32_t& compaction_block_size
	) {
		free_allocations();

		batch_size = tcnn::next_multiple(n_elements_per_batch, tcnn::batch_size_granularity);
		uint32_t n_output_pixel_elements = tcnn::next_multiple(4 * output_width * output_height, tcnn::batch_size_granularity);

		// camera
		camera			= allocate<Camera>(stream, 1);
		bounding_box	= allocate<BoundingBox>(stream, 1);
		occupancy_grid	= allocate<CascadedOccupancyGrid>(stream, 1);

		// compaction
		c_block_size	= compaction_block_size;
		c_n_blocks		= tcnn::div_round_up((int)batch_size, c_block_size);

		c_block_counts	= allocate<int>(stream, c_n_blocks);
		c_block_offsets	= allocate<int>(stream, c_n_blocks);
		compact_idx		= allocate<int>(stream, batch_size);

		// rays
		ray_alive		= allocate<bool>(stream, batch_size); // no need to double buffer

		// double buffers
		ray_active[0]	= allocate<bool>(stream, 2 * batch_size);
		ray_active[1]	= ray_active[0] + batch_size;

		ray_origin[0]	= allocate<float>(stream, 2 * 3 * batch_size);
		ray_origin[1]	= ray_origin[0] + 3 * batch_size;
		
		ray_dir[0]		= allocate<float>(stream, 2 * 3 * batch_size);
		ray_dir[1]		= ray_dir[0] + 3 * batch_size;

		ray_idir[0]		= allocate<float>(stream, 2 * 3 * batch_size);
		ray_idir[1]		= ray_idir[0] + 3 * batch_size;

		ray_t[0]		= allocate<float>(stream, 2 * batch_size);
		ray_t[1]		= ray_t[0] + batch_size;

		ray_idx[0]		= allocate<uint32_t>(stream, 2 * batch_size);
		ray_idx[1]		= ray_idx[0] + batch_size;

		ray_sigma[0]	= allocate<float>(stream, 2 * batch_size);
		ray_sigma[1]	= ray_sigma[0] + batch_size;

		// samples
		sample_pos		= allocate<float>(stream, 3 * batch_size);
		sample_dt		= allocate<float>(stream, batch_size);

		// network
		network_sigma	= allocate<tcnn::network_precision_t>(stream, n_network_sigma_elements * batch_size);
		network_color	= allocate<tcnn::network_precision_t>(stream, n_network_color_elements * batch_size);

		// output
		pixel_buffer	= allocate<float>(stream, n_output_pixel_elements);
	};
};

NRC_NAMESPACE_END