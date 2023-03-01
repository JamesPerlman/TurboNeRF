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

using namespace nrc;
using namespace tcnn;
using namespace nlohmann;


void Trainer::create_pixel_undistort_map(
	Trainer::Context& ctx
) {
	const Camera& camera = ctx.dataset->cameras[0];
	const uint32_t w = camera.resolution.x;
	const uint32_t h = camera.resolution.y;

	const uint32_t n_pixels = w * h;

	const dim3 block(16, 16);
	const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

	// create the undistort map for camera 0 - assumption: all cameras have identical dist_params params
	generate_undistorted_pixel_map_kernel<<<next_multiple(n_pixels, batch_size_granularity), n_threads_linear, 0, ctx.stream>>>(
		n_pixels,
		camera,
		ctx.workspace.undistort_map
	);
}

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

	curandStatus_t status = curandGenerateUniform(ctx.rng, ctx.workspace.random_float, ctx.workspace.batch_size);
	if (status != CURAND_STATUS_SUCCESS) {
		throw std::runtime_error("Error generating random floats for training batch.");
	}
	
	/**
	 * Generate rays and pixels for training
	 * 
	 * We can take a shortcut here and generate only the data needed to fill the batch back up.
	 * If not all the previous batch's rays were used, then we only need to regenerate rays
	 * for batch_size minus the number of rays that were used.
	 */

	const float n_rays_per_image = static_cast<float>(ctx.n_rays_in_batch) / static_cast<float>(ctx.dataset->images.size());
	const float chunk_size = static_cast<float>(ctx.dataset->n_pixels_per_image) / n_rays_per_image;

	initialize_training_rays_and_pixels_kernel<<<n_blocks_linear(ctx.n_rays_in_batch), n_threads_linear, 0, ctx.stream>>>(
		ctx.n_rays_in_batch,
		ctx.workspace.batch_size,
		ctx.dataset->images.size(),
		ctx.dataset->n_pixels_per_image,
		ctx.dataset->n_pixels_per_image * ctx.dataset->n_channels_per_image,
		ctx.dataset->image_dimensions,
		n_rays_per_image,
		chunk_size,
		ctx.workspace.bounding_box,

		// input buffers
		ctx.workspace.cameras.data(),
		ctx.workspace.undistort_map,
		ctx.workspace.image_data,
		ctx.workspace.random_float,

		// output buffers
		ctx.workspace.pix_rgba,
		ctx.workspace.ray_origin,
		ctx.workspace.ray_dir,
		ctx.workspace.ray_inv_dir,
		ctx.workspace.ray_t,
		ctx.workspace.ray_alive
	);

	const float dt_min = NeRFConstants::min_step_size;
	const float dt_max = ctx.dataset->bounding_box.size_x * dt_min;
	const float cone_angle = NeRFConstants::cone_angle;

	// Count the number of steps each ray would take.  We only need to do this for the new rays.
	march_and_count_steps_per_ray_kernel<<<n_blocks_linear(ctx.n_rays_in_batch), n_threads_linear, 0, ctx.stream>>>(
		ctx.n_rays_in_batch,
		ctx.workspace.batch_size,
		ctx.workspace.bounding_box,
		ctx.workspace.occupancy_grid,
		cone_angle,
		dt_min,
		dt_max,

		// output buffers
		ctx.workspace.ray_dir,
		ctx.workspace.ray_inv_dir,
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
		ctx.workspace.bounding_box,
		1.0f / ctx.dataset->bounding_box.size_x,
		ctx.workspace.occupancy_grid,
		dt_min,
		dt_max,
		cone_angle,

		// input buffers
		ctx.workspace.random_float,
		ctx.workspace.ray_origin,
		ctx.workspace.ray_dir,
		ctx.workspace.ray_inv_dir,
		ctx.workspace.ray_t,
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

void Trainer::update_occupancy_grid(
	Trainer::Context& ctx,
	const float& selection_threshold
) {
	const uint32_t grid_volume = ctx.nerf->occupancy_grid.volume_i;
	const uint32_t n_bitfield_bytes = ctx.nerf->occupancy_grid.get_n_bitfield_elements();
	const uint32_t n_levels = ctx.nerf->occupancy_grid.n_levels;
	const float inv_aabb_size = 1.0f / ctx.nerf->bounding_box.size_x;

	// loop through each grid level, querying the network for the density at each cell and updating the occupancy grid's density
	for (int level = 0; level < n_levels; ++level) {

		// update occupancy grid values
		uint32_t n_cells_updated = 0;
		while (n_cells_updated < grid_volume) {
			uint32_t n_cells_to_update = std::min(grid_volume - n_cells_updated, ctx.workspace.batch_size);

			uint32_t batch_size = tcnn::next_multiple(n_cells_to_update, tcnn::batch_size_granularity);
			
			// generate random floats for sampling
			curandStatus_t status = curandGenerateUniform(ctx.rng, ctx.workspace.random_float, 4 * batch_size);
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
			ctx.network.inference(
				ctx.stream,
				ctx.nerf->params,
				batch_size,
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
				NeRFConstants::occupancy_decay,
				selection_threshold,
				ctx.workspace.random_float + 3 * batch_size, // (random_float + 3 * batch_size) is so thresholding doesn't correspond to x,y,z positions
				ctx.workspace.network_output + 3 * batch_size,
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

	CHECK_DATA(bitfield_cpu, uint8_t, ctx.nerf->occupancy_grid.get_bitfield(), n_bitfield_bytes, ctx.stream);

	// this is just debug code to print out the percentage of bits occupied
	int bits_occupied = 0;
	for (int i = 0; i < n_bitfield_bytes; ++i) {
		for (int j = 0; j < n_levels; ++j) {
			if (bitfield_cpu[i] & (1 << j)) {
				++bits_occupied;
			}
		}
	}

	printf("%% of bits occupied: %f\n", 100.f * (float)bits_occupied / (grid_volume * n_levels));
	
	CHECK_DATA(cpu_ogrid_dens, float, ctx.nerf->occupancy_grid.get_density(), grid_volume, ctx.stream);
	float min_value = std::numeric_limits<float>::max();
	float max_value = std::numeric_limits<float>::lowest();
	float sum_value = 0.0f;
	for (int i = 0; i < grid_volume; ++i) {
		if (cpu_ogrid_dens[i] < min_value) min_value = cpu_ogrid_dens[i];
		if (cpu_ogrid_dens[i] > max_value) max_value = cpu_ogrid_dens[i];
		sum_value += cpu_ogrid_dens[i];
	}

	float avg_value = sum_value / grid_volume;
	printf("Occupancy Grid: min = %f, max = %f, avg = %f\n", min_value, max_value, avg_value);
}

void Trainer::train_step(
	Trainer::Context& ctx
) {
	// Generate training batch
	generate_next_training_batch(ctx);

	ctx.network.train(
		ctx.stream,
		ctx.nerf->params,
		ctx.workspace.batch_size,
		ctx.n_rays_in_batch,
		ctx.n_samples_in_batch,
		ctx.workspace.ray_step,
		ctx.workspace.ray_offset,
		ctx.workspace.sample_pos,
		ctx.workspace.sample_dir,
		ctx.workspace.sample_dt,
		ctx.workspace.pix_rgba,
		ctx.workspace.network_concat,
		ctx.workspace.network_output
	);
}
