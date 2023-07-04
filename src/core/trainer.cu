#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <json/json.hpp>
#include <thrust/device_vector.h>
#include <tiny-cuda-nn/common.h>

#include "trainer.cuh"

#include "../utils/camera-kernels.cuh"
#include "../utils/nerf-constants.cuh"
#include "../utils/occupancy-grid-kernels.cuh"
#include "../utils/parallel-utils.cuh"
#include "../utils/stream-compaction.cuh"
#include "../utils/training-batch-kernels.cuh"

#include "../common.h"

#include <iostream>
#include <fstream>

using namespace turbo;
using namespace tcnn;
using namespace nlohmann;

/**
 * generate_next_training_batch does the following:
  * 
  * 0. Generate rays and ground truth pixels from training cameras and images
  * 1. Count the number of steps each ray will take
  * 2. Determine the maximum number of rays that will fill the batch with samples
  * 3. Generate the samples for the density network
  * 4. Query the network, determine:
  * 	4a. Which samples are visible?
  * 	4b. What is the transmittance of each sample?
  * 5. Compact this batch of samples, excluding the invisible ones
  * 6. Repeat steps 0-5 until the batch is full enough
 */

void Trainer::generate_next_training_batch(
	Trainer::Context& ctx
) {
	
	// Generate random floats for use in training
	// increment the seed by the number of floats generated
	CURAND_ASSERT_SUCCESS(
		curandGenerateUniform(ctx.rng, ctx.workspace.random_float, 4 * ctx.workspace.batch_size)
	);

	ctx.rng_offset += (unsigned long long)(4 * ctx.workspace.batch_size);
	CURAND_ASSERT_SUCCESS(curandSetGeneratorOffset(ctx.rng, ctx.rng_offset));
	
	/**
	 * Generate rays and pixels for training
	 * 
	 * We can take a shortcut here and generate only the data needed to fill the batch back up.
	 * If not all the previous batch's rays were used, then we only need to regenerate rays
	 * for batch_size minus the number of rays that were used.
	 */

	const float n_rays_per_image = static_cast<float>(ctx.n_rays_in_batch) / static_cast<float>(ctx.dataset->images.size());

	initialize_training_rays_and_pixels_kernel<<<n_blocks_linear(ctx.n_rays_in_batch), n_threads_linear, 0, ctx.stream>>>(
		ctx.n_rays_in_batch,
		ctx.workspace.batch_size,
		ctx.dataset->images.size(),
		ctx.dataset->n_pixels_per_image,
		ctx.dataset->image_dimensions,
		n_rays_per_image,

		// input buffers
		ctx.nerf->dataset_ws.bounding_box,
		ctx.nerf->dataset_ws.cameras,
		ctx.nerf->dataset_ws.image_data,
		ctx.workspace.random_float,

		// output buffers
		ctx.workspace.pix_rgba,
		ctx.workspace.ray_origin,
		ctx.workspace.ray_dir,
		ctx.workspace.ray_t,
		ctx.workspace.ray_t_max,
		ctx.workspace.ray_alive
	);

	if (ctx.alpha_selection_threshold < 1.0f && ctx.alpha_selection_probability < 1.0f) {
		deactivate_rays_with_alpha_threshold_kernel<<<n_blocks_linear(ctx.n_rays_in_batch), n_threads_linear, 0, ctx.stream>>>(
			ctx.n_rays_in_batch,
			ctx.workspace.batch_size,
			ctx.alpha_selection_threshold,
			ctx.alpha_selection_probability,

			// input buffers
			ctx.workspace.pix_rgba,
			ctx.workspace.random_float + ctx.workspace.batch_size,

			// output buffer
			ctx.workspace.ray_alive
		);
	}

	const float dt_min = ctx.min_step_size;
	const float dt_max = ctx.dataset->bounding_box.size() * dt_min;
	const float cone_angle = NeRFConstants::cone_angle;

	// Count the number of steps each ray would take.  We only need to do this for the new rays.
	march_and_count_steps_per_ray_kernel<<<n_blocks_linear(ctx.n_rays_in_batch), n_threads_linear, 0, ctx.stream>>>(
		ctx.n_rays_in_batch,
		ctx.workspace.batch_size,
		ctx.nerf->dataset_ws.bounding_box,
		ctx.workspace.occupancy_grid,
		cone_angle,
		dt_min,
		dt_max,

		// input buffers
		ctx.workspace.ray_dir,
		ctx.workspace.ray_t_max,

		// output buffers
		ctx.workspace.ray_alive,
		ctx.workspace.ray_origin,
		ctx.workspace.ray_t,
		ctx.workspace.ray_step
	);

	/**
	 * Count the number of rays that will fill the batch with the maximum number of samples
	 * 
	 * We don't know how many rays will be in the batch, but we can make an educated guess to limit the amount of computation time by the exclusive_scan.
	 */
	
	uint32_t n_estimated_rays = 0;

	if (ctx.n_samples_in_batch > 0) {
		const float mean_rays_per_sample = static_cast<float>(ctx.n_rays_in_batch) / static_cast<float>(ctx.n_samples_in_batch);
		n_estimated_rays = static_cast<uint32_t>(mean_rays_per_sample * static_cast<float>(ctx.workspace.batch_size));
	} else {
		n_estimated_rays = 16384;
	}

	// Grab some references to the n_steps arrays
	thrust::device_ptr<uint32_t> n_steps_ptr(ctx.workspace.ray_step);
	thrust::device_ptr<uint32_t> ray_offset_ptr(ctx.workspace.ray_offset);
	
	// get ray memory offsets
	thrust::exclusive_scan(
		thrust::cuda::par.on(ctx.stream),
		n_steps_ptr,
		n_steps_ptr + n_estimated_rays,
		ray_offset_ptr
	);

	// Count the number of rays actually used to fill the sample batch
	const int n_ray_max_idx = find_last_lt_presorted(
		ctx.stream,
		ray_offset_ptr,
		n_estimated_rays,
		ctx.workspace.batch_size
	) - 1;

	if (n_ray_max_idx < 0) {
		throw std::runtime_error("No rays were generated for this training batch!\n");
	}

	ctx.n_rays_in_batch = static_cast<uint32_t>(n_ray_max_idx + 1);

	// Count the number of samples that will be generated
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(
			&ctx.n_samples_in_batch,
			ray_offset_ptr.get() + ctx.n_rays_in_batch,
			sizeof(uint32_t),
			cudaMemcpyDeviceToHost,
			ctx.stream
		)
	);

	CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));

	if (ctx.n_samples_in_batch < 1) {
		throw std::runtime_error("No samples were generated for this training batch!\n");
	}

	// Generate sample positions
	march_and_generate_network_positions_kernel<<<n_blocks_linear(ctx.n_rays_in_batch), n_threads_linear, 0, ctx.stream>>>(
		ctx.n_rays_in_batch,
		ctx.workspace.batch_size,
		ctx.nerf->dataset_ws.bounding_box,
		1.0f / ctx.dataset->bounding_box.size(),
		ctx.workspace.occupancy_grid,
		dt_min,
		dt_max,
		cone_angle,

		// input buffers
		ctx.workspace.ray_origin,
		ctx.workspace.ray_dir,
		ctx.workspace.ray_t,
		ctx.workspace.ray_t_max,
		ctx.workspace.ray_offset,
		ctx.workspace.ray_alive,

		// dual-use buffers
		ctx.workspace.ray_step,

		// output buffers
		ctx.workspace.sample_pos,
		ctx.workspace.sample_dir,
		ctx.workspace.sample_dt
	);
}

// update occupancy grid

uint32_t Trainer::update_occupancy_grid(Trainer::Context& ctx, const uint32_t& training_step) {
	const auto& proxy = ctx.nerf->proxy;
	const uint32_t grid_volume = ctx.nerf->occupancy_grid.volume_i;
	const uint32_t n_bitfield_bytes = ctx.nerf->occupancy_grid.get_n_bitfield_elements();
	const uint32_t n_levels = ctx.nerf->occupancy_grid.n_levels;
	const float aabb_size = proxy->training_bbox.get().size();
	const float inv_aabb_size = 1.0f / aabb_size;

	// decay occupancy grid values
	decay_occupancy_grid_values_kernel<<<n_blocks_linear(grid_volume), n_threads_linear, 0, ctx.stream>>>(
		ctx.workspace.occupancy_grid,
		NeRFConstants::occupancy_decay
	);

	// loop through each grid level, querying the network for the density at each cell and updating the occupancy grid's density
	for (int level = 0; level < n_levels; ++level) {

		// update occupancy grid values
		uint32_t n_cells_updated = 0;
		while (n_cells_updated < grid_volume) {
			uint32_t n_cells_to_update = std::min(grid_volume - n_cells_updated, ctx.workspace.batch_size);

			uint32_t batch_size = tcnn::next_multiple(n_cells_to_update, tcnn::batch_size_granularity);
			
			// generate random floats for sampling
			curandStatus_t status = curandGenerateUniform(ctx.rng, ctx.workspace.random_float, 4 * batch_size);
			ctx.rng_offset += (unsigned long long)(4 * ctx.workspace.batch_size);
			curandSetGeneratorOffset(ctx.rng, ctx.rng_offset);

			if (status != CURAND_STATUS_SUCCESS) {
				printf("Error generating random floats for occupancy grid update.\n");
			}

			// generate random sampling points
			generate_grid_cell_network_sample_points_kernel<<<n_blocks_linear(n_cells_to_update), n_threads_linear, 0, ctx.stream>>>(
				n_cells_to_update,
				batch_size,
				n_cells_updated,
				ctx.workspace.occupancy_grid,
				level,
				inv_aabb_size,
				ctx.workspace.random_float,
				ctx.workspace.sample_pos
			);

			// query the density network
			ctx.nerf->network.inference(
				ctx.stream,
				ctx.nerf->params,
				batch_size,
				aabb_size,
				ctx.workspace.sample_pos,
				nullptr,
				ctx.workspace.network_concat,
				ctx.workspace.network_output,
				false
			);

			// update occupancy grid values
			update_occupancy_with_density_kernel<<<n_blocks_linear(n_cells_to_update), n_threads_linear, 0, ctx.stream>>>(
				n_cells_to_update,
				n_cells_updated,
				level,
				training_step < 256,
				ctx.workspace.random_float + 3 * batch_size, // (random_float + 3 * batch_size) is so thresholding doesn't correspond to x,y,z positions
				ctx.workspace.network_concat,
				ctx.workspace.occupancy_grid
			);

			n_cells_updated += n_cells_to_update;
		}
	}

	// update the bits by thresholding the density values

	update_occupancy_grid_bits_kernel<<<n_blocks_linear(n_bitfield_bytes), n_threads_linear, 0, ctx.stream>>>(
		NeRFConstants::occupancy_threshold,
		ctx.workspace.occupancy_grid
	);

	uint32_t n_bits_occupied = count_1s(
		ctx.nerf->occupancy_grid.get_bitfield(),
		ctx.nerf->occupancy_grid.get_bitcounts(),
		n_bitfield_bytes,
		ctx.stream
	);

	return n_bits_occupied;
}

float Trainer::train_step(
	Trainer::Context& ctx,
	const uint32_t& training_step
) {
	// Generate training batch
	generate_next_training_batch(ctx);

	const float loss = ctx.nerf->network.train(
		ctx.stream,
		ctx.nerf->params,
		training_step,
		ctx.workspace.batch_size,
		ctx.n_rays_in_batch,
		ctx.n_samples_in_batch,
		ctx.nerf->proxy->training_bbox.get().size(),
		ctx.workspace.random_float + ctx.workspace.batch_size,
		ctx.workspace.ray_step,
		ctx.workspace.ray_offset,
		ctx.workspace.sample_pos,
		ctx.workspace.sample_dir,
		ctx.workspace.sample_dt,
		ctx.workspace.pix_rgba,
		ctx.workspace.network_concat,
		ctx.workspace.network_output
	);

	return loss;
}
