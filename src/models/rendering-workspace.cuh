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

	int* compact_idx;

	// rays
	bool* ray_alive;
	bool* ray_active;

	float* ray_origin;
	float* ray_dir;
	float* ray_idir;
	float* ray_t;

	// 2D ray index (x, y)
	uint32_t* ray_idx; 
	
	uint32_t* ray_steps;
	uint32_t* ray_steps_cum;

	uint32_t* ray_x;
	uint32_t* ray_y;
	
	// samples
	float* sample_pos;
	float* sample_dir;
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
		const uint32_t& n_network_color_elements
	) {
		free_allocations();

		batch_size = tcnn::next_multiple(n_elements_per_batch, tcnn::batch_size_granularity);
		uint32_t n_output_pixel_elements = tcnn::next_multiple(4 * output_width * output_height, tcnn::batch_size_granularity);

		// camera
		camera			= allocate<Camera>(stream, 1);
		bounding_box	= allocate<BoundingBox>(stream, 1);
		occupancy_grid	= allocate<CascadedOccupancyGrid>(stream, 1);

		compact_idx		= allocate<int>(stream, batch_size);

		// rays
		ray_alive		= allocate<bool>(stream, batch_size);
		ray_active		= allocate<bool>(stream, batch_size);
		ray_origin		= allocate<float>(stream, 3 * batch_size);
		ray_dir			= allocate<float>(stream, 3 * batch_size);
		ray_idir 		= allocate<float>(stream, 3 * batch_size);
		ray_t			= allocate<float>(stream, batch_size);
		ray_steps		= allocate<uint32_t>(stream, batch_size);
		ray_steps_cum	= allocate<uint32_t>(stream, batch_size);
		ray_idx			= allocate<uint32_t>(stream, batch_size);

		// samples
		sample_pos		= allocate<float>(stream, 3 * batch_size);
		sample_dir		= allocate<float>(stream, 3 * batch_size);
		sample_dt		= allocate<float>(stream, batch_size);

		// network
		network_sigma	= allocate<tcnn::network_precision_t>(stream, n_network_sigma_elements * batch_size);
		network_color	= allocate<tcnn::network_precision_t>(stream, n_network_color_elements * batch_size);

		// output
		pixel_buffer	= allocate<float>(stream, n_output_pixel_elements);
	};
};

NRC_NAMESPACE_END