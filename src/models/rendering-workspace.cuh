#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "camera.cuh"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct RenderingWorkspace: Workspace {

	uint32_t batch_size;
	
	const Camera* camera;

	// rays
	bool* ray_alive;
	bool* ray_active;

	float* ray_origin;
	float* ray_dir;
	float* ray_t;
	
	uint32_t* ray_steps;
	uint32_t* ray_steps_cum;

	uint32_t* ray_x;
	uint32_t* ray_y;
	
	// samples
	float* sample_pos;
	float* sample_dir;
	float* sample_dt;

	// output buffers
	tcnn::network_precision_t* network_output;
	float* output_buffer;

	// samples
	void enlarge(
		const cudaStream_t& stream,
		const uint32_t& output_width,
		const uint32_t& output_height,
		const uint32_t& n_rays_per_batch
	) {
		free_allocations();
		
		batch_size = tcnn::next_multiple(batch_size, tcnn::batch_size_granularity);
		uint32_t n_output_pixel_elements = tcnn::next_multiple(4 * output_width * output_height, tcnn::batch_size_granularity);

		// rays
		ray_alive		= allocate<bool>(stream, batch_size);
		ray_active		= allocate<bool>(stream, batch_size);
		ray_origin		= allocate<float>(stream, 3 * batch_size);
		ray_dir			= allocate<float>(stream, 3 * batch_size);
		ray_t			= allocate<float>(stream, batch_size);
		ray_steps		= allocate<uint32_t>(stream, batch_size);
		ray_steps_cum	= allocate<uint32_t>(stream, batch_size);

		// samples
		sample_pos		= allocate<float>(stream, 3 * batch_size);
		sample_dir		= allocate<float>(stream, 3 * batch_size);
		sample_dt		= allocate<float>(stream, batch_size);

		output_buffer	= allocate<float>(stream, n_output_pixel_elements);
	};
};

NRC_NAMESPACE_END