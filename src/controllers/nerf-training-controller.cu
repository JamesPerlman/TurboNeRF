
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <json/json.hpp>
#include <thrust/device_vector.h>
#include <tiny-cuda-nn/common.h>


#include "nerf-training-controller.h"

#include "../utils/stream-compaction.cuh"
#include "../utils/nerf-constants.cuh"
#include "../utils/occupancy-grid-kernels.cuh"
#include "../utils/parallel-utils.cuh"
#include "../utils/training-batch-kernels.cuh"

#include "../common.h"

#include <iostream>
#include <fstream>

using namespace nrc;
using namespace tcnn;
using namespace nlohmann;

NeRFTrainingController::NeRFTrainingController(Dataset& dataset, NeRF* nerf)
	: dataset(dataset), nerf(nerf)
{	
	// TODO: refactor size_x to just size?
	// RNG
	// todo: CURAND_ASSERT_SUCCESS
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandGenerateSeeds(rng);

}

NeRFTrainingController::~NeRFTrainingController() {
	curandDestroyGenerator(rng);
}

// NeRFTrainingController member functions

void NeRFTrainingController::prepare_for_training(
	const cudaStream_t& stream,
	const uint32_t& batch_size
) {
	// This allocates memory for all the elements we need during training
	workspace.enlarge(stream,
		dataset.images.size(),
		dataset.n_pixels_per_image,
		dataset.n_channels_per_image,
		batch_size,
		nerf->occupancy_grid.n_levels,
		nerf->occupancy_grid.resolution_i,
		nerf->network.get_concat_buffer_width(),
		nerf->network.get_padded_output_width()
	);

	// Create a CascadedOccupancyGrid object and copy it to the GPU
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(workspace.occ_grid, &nerf->occupancy_grid, sizeof(CascadedOccupancyGrid), cudaMemcpyHostToDevice, stream)
	);

	// Copy dataset's BoundingBox to the GPU
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(workspace.bounding_box, &dataset.bounding_box, sizeof(BoundingBox), cudaMemcpyHostToDevice, stream)
	);

	// Training image indices will be reused for each batch.  We select the same number of rays from each image in the dataset.
	generate_training_image_indices<<<n_blocks_linear(workspace.batch_size), n_threads_linear, 0, stream>>>(
		workspace.batch_size,
		dataset.images.size(),
		workspace.img_index
	);
	
	// Copy training cameras to the GPU
	workspace.cameras.resize_and_copy_from_host(dataset.cameras);

	// Load all images into GPU memory!
	load_images(stream);

	// Since there is no previous step here, we set the number of previous rays to the batch size
	// so that the training batch generator will generate a full batch of rays
	n_rays_in_batch = workspace.batch_size;
	training_step = 0;

	// Initialize the network
	nerf->network.prepare_for_training(stream);
}

void NeRFTrainingController::load_images(const cudaStream_t& stream) {
	// make sure images are all loaded into CPU and GPU
	size_t n_image_elements = dataset.n_channels_per_image * dataset.n_pixels_per_image;
	size_t image_size = n_image_elements * sizeof(stbi_uc);

	dataset.load_images_in_parallel(
		[this, &image_size, &n_image_elements, &stream](const size_t& image_index, const TrainingImage& image) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(
				workspace.image_data + image_index * n_image_elements,
				image.data_cpu.get(),
				image_size,
				cudaMemcpyHostToDevice,
				stream
			));
		}
	);

	printf("All images loaded to GPU.\n");
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

void NeRFTrainingController::generate_next_training_batch(cudaStream_t stream) {

	// Generate random floats for use in training

	curandStatus_t status = curandGenerateUniform(rng, workspace.random_float, workspace.batch_size);
	if (status != CURAND_STATUS_SUCCESS) {
		printf("Error generating random floats for training batch.\n");
	}
	
	// Convert floats to uint32_t which will be interpreted as pixel indices for any training image
	resize_floats_to_uint32_with_max<<<n_blocks_linear(workspace.batch_size), n_threads_linear, 0, stream>>>(
		workspace.batch_size, workspace.random_float, workspace.pix_index, dataset.n_pixels_per_image
	);

	/**
	 * Generate rays and pixels for training
	 * 
	 * We can take a shortcut here and generate only the data needed to fill the batch back up.
	 * If not all the previous batch's rays were used, then we only need to regenerate rays
	 * for batch_size minus the number of rays that were used.
	 */

	initialize_training_rays_and_pixels_kernel<<<n_blocks_linear(n_rays_in_batch), n_threads_linear, 0, stream>>>(
		n_rays_in_batch,
		workspace.batch_size,
		dataset.images.size(),
		dataset.n_pixels_per_image * dataset.n_channels_per_image,
		dataset.image_dimensions,
		workspace.bounding_box,
		workspace.cameras.data(),
		workspace.image_data,
		workspace.img_index,
		workspace.pix_index,
		workspace.pix_rgba[0],
		workspace.ray_origin,
		workspace.ray_dir[0],
		workspace.ray_inv_dir,
		workspace.ray_t,
		workspace.ray_alive
	);

	// Count the number of steps each ray will take
	
	const float dt_min = NeRFConstants::min_step_size;
	const float dt_max = dataset.bounding_box.size_x * dt_min;
	const float cone_angle = NeRFConstants::cone_angle;

	// Count the number of steps each ray would take.  We only need to do this for the new rays.
	march_and_count_steps_per_ray_kernel<<<n_blocks_linear(n_rays_in_batch), n_threads_linear, 0, stream>>>(
		n_rays_in_batch,
		workspace.batch_size,
		workspace.bounding_box,
		workspace.occ_grid,
		cone_angle,
		dt_min,
		dt_max,
		workspace.ray_dir[0],
		workspace.ray_inv_dir,
		workspace.ray_alive,
		workspace.ray_origin,
		workspace.ray_t,
		workspace.ray_step[0]
	);


	/*
	 * Now we loop through the rays, generate samples for the density network, filter out the invisible ones, and compact the batch.
	 * We do this until the sample batch is full enough.
	 */
	uint32_t n_batch_samples = 0; // This is the number of samples we have generated for the batch so far
	uint32_t min_sample_batch = 0; //1024; // This is the smallest minibatch of samples that is worth running on the GPU.
	uint32_t n_batch_rays = 0; // this is used as an offset to where new ray step counts should be written to in the ray_step buffer
	uint32_t n_rays_used = 0; // this is the number of rays we have used from the buffers created above

	/*
	 * There are two main terminating conditions for this loop:
	 * 1. The batch is full enough
	 * 2. We've used up all the rays in the batch
	 * 
	 * In the second case, there simply aren't enough samples to fill the whole batch.  I guess that is ok.
	 * 
	 * And one condition that would cause us to break out of the loop early:
	 * 3. If the number of steps for a single ray exceeds the batch size, we break.
	 * 
	 * There are two continue-conditions inside the loop that we should be aware of:
	 * 1. If no samples were generated for the entire batch of rays, we skip the rest of the loop.
	 * 2. If samples were generated but it was discovered they are not visible, we also continue.
	 * 
	 * In the second case, it's not clear if this can actually happen.
	 * 
	 */
	int n_iters = 0;
	while (n_batch_samples < workspace.batch_size - min_sample_batch && n_rays_used < workspace.batch_size) {
		const uint32_t current_ray_offset = n_rays_used;

		/**
		 * Cumulative summation via inclusive_scan gives us the offset index that each ray's first sample should start at, relative to the start of the batch.
		 * We need to perform this cumsum over the entire batch of rays, not just the rays that were regenerated over the used ones in the previous batch.
		 */

		// Grab some references to the n_steps arrays
		thrust::device_ptr<uint32_t> n_uncompacted_steps_ptr(workspace.ray_step[0]);
		thrust::device_ptr<uint32_t> n_uncompacted_steps_cum_ptr(workspace.ray_step_cum[0]);
		
		// cumulative sum the number of steps for each ray
		thrust::inclusive_scan(
			thrust::cuda::par.on(stream),
			n_uncompacted_steps_ptr + current_ray_offset,
			n_uncompacted_steps_ptr + workspace.batch_size,
			n_uncompacted_steps_cum_ptr
		);

		CHECK_DATA(n_steps_cpu, uint32_t, workspace.ray_step[0], workspace.batch_size);
		CHECK_DATA(n_steps_cum_cpu, uint32_t, workspace.ray_step_cum[0], workspace.batch_size);
		// Count the number of rays actually used to fill the sample batch
		const int n_ray_max_idx = find_last_lt_presorted(
			stream,
			n_uncompacted_steps_cum_ptr,
			workspace.batch_size - current_ray_offset,
			workspace.batch_size - n_batch_samples
		);

		// This condition breaks out of the loop if the number of steps for a single ray exceeds the remaining batch slots available
		if (n_ray_max_idx < 0) {
			break;
		}

		const uint32_t n_uncompacted_rays = static_cast<uint32_t>(n_ray_max_idx + 1);
		n_rays_used += n_uncompacted_rays;

		// Copy the number of samples that will be generated for all rays until the last ray, to the host
		uint32_t n_uncompacted_samples = 0;

		CUDA_CHECK_THROW(
			cudaMemcpyAsync(
				&n_uncompacted_samples,
				n_uncompacted_steps_cum_ptr.get() + n_ray_max_idx,
				sizeof(uint32_t),
				cudaMemcpyDeviceToHost,
				stream
			)
		);

		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		// early continue if no samples will be generated
		if (n_uncompacted_samples == 0) {
			continue;
		}

		// March (again), this time generating position samples for the density network 
		march_and_generate_network_positions_kernel<<<n_blocks_linear(n_uncompacted_rays), n_threads_linear, 0, stream>>>(
			n_uncompacted_rays,
			workspace.batch_size,
			workspace.bounding_box,
			1.0f / dataset.bounding_box.size_x,
			workspace.occ_grid,
			dt_min,
			dt_max,
			cone_angle,

			// input buffers
			workspace.ray_origin	+ current_ray_offset,
			workspace.ray_dir[0]	+ current_ray_offset,
			workspace.ray_inv_dir	+ current_ray_offset,
			workspace.ray_t			+ current_ray_offset,
			workspace.ray_step_cum[0],
			workspace.ray_alive		+ current_ray_offset,

			// dual-use buffers
			workspace.ray_step[0]	+ current_ray_offset,

			// output buffers
			workspace.sample_pos[0],
			workspace.sample_dt[0]
		);

		// run these samples through the density network and determine their visibility
		nerf->network.sample_density_and_count_steps(
			stream,
			n_uncompacted_rays,
			n_uncompacted_samples,
			workspace.batch_size,
			NeRFConstants::min_transmittance,

			// input buffers
			workspace.ray_step_cum[0],
			workspace.sample_pos[0],
			workspace.sample_dt[0],

			// dual-use buffers
			workspace.ray_step[0] + current_ray_offset,

			// output buffers
			workspace.network_concat // just use network_concat as a temporary buffer.  nobody else is using it right now.
		);
		
		// compact ray steps by excluding the rays that have no visible samples
		uint32_t n_visible_rays = count_nonzero_elements(
			stream,
			n_uncompacted_rays,
			workspace.ray_step[0] + current_ray_offset
		);

		if (n_visible_rays == 0) {
			continue;
		}

		generate_nonzero_compaction_indices(
			stream,
			n_uncompacted_rays,
			workspace.ray_step[0] + current_ray_offset,
			workspace.ray_index
		);

		compact_ray_buffers_kernel<<<n_blocks_linear(n_visible_rays), n_threads_linear, 0, stream>>>(
			n_visible_rays,
			workspace.batch_size,
			workspace.ray_index,
			workspace.ray_step[0]	+ current_ray_offset,
			workspace.pix_rgba[0]	+ current_ray_offset,
			workspace.ray_step[1]	+ n_batch_rays,
			workspace.pix_rgba[1]	+ n_batch_rays
		);

		// cumulative sum compacted ray steps to get the compaction offsets for the samples
		// TODO: if we can give this an initial value of the element at index current_ray_offset in the previous cumsum we can avoid recalculating the whole cumsum
		thrust::device_ptr<uint32_t> n_compacted_steps_ptr(workspace.ray_step[1]);
		thrust::device_ptr<uint32_t> n_compacted_steps_cum_ptr(workspace.ray_step_cum[1]);

		thrust::inclusive_scan(
			thrust::cuda::par.on(stream),
			n_compacted_steps_ptr + n_batch_rays,
			n_compacted_steps_ptr + n_batch_rays + n_visible_rays,
			n_compacted_steps_cum_ptr
		);

		// find the number of compacted samples
		const int n_compacted_ray_max_idx = find_last_lt_presorted(
			stream,
			n_compacted_steps_cum_ptr,
			n_visible_rays,
			workspace.batch_size - n_batch_samples
		);

		// we don't need to break check the n_compacted_ray_max_idx because we know there is enough space from the uncompacted ray check earlier
		uint32_t n_visible_samples = 0;

		CUDA_CHECK_THROW(
			cudaMemcpyAsync(
				&n_visible_samples,
				n_compacted_steps_cum_ptr.get() + n_compacted_ray_max_idx,
				sizeof(uint32_t),
				cudaMemcpyDeviceToHost,
				stream
			)
		);

		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		// we do need to check if any compacted samples will be generated
		if (n_visible_samples == 0) {
			continue;
		}

		// compact buffers
		compact_sample_buffers_kernel<<<n_blocks_linear(n_visible_rays), n_threads_linear, 0, stream>>>(
			n_visible_rays,
			workspace.batch_size,

			// input buffers
			workspace.ray_index,
			workspace.ray_step[1]	+ n_batch_rays,
			workspace.ray_step_cum[0],
			workspace.ray_step_cum[1],
			workspace.ray_dir[0]	+ current_ray_offset,
			workspace.sample_pos[0],
			workspace.sample_dt[0],

			// output buffers
			workspace.sample_dir	+ n_batch_samples,
			workspace.sample_pos[1]	+ n_batch_samples,
			workspace.sample_dt[1]	+ n_batch_samples
		);

		n_batch_samples += n_visible_samples;
		n_batch_rays += n_visible_rays;
		++n_iters;
	}

	printf("n_iters: %d\n", n_iters);

	if (n_batch_samples == 0) {
		throw std::runtime_error("No samples were generated for this training batch!\n");
	}

	// one final time, we need to count the steps in all rays and get a cumulative sum
	thrust::device_ptr<uint32_t> n_steps_in_ptr(workspace.ray_step[1]);

	thrust::inclusive_scan(
		thrust::cuda::par.on(stream),
		n_steps_in_ptr,
		n_steps_in_ptr + n_batch_rays,
		thrust::device_ptr<uint32_t>(workspace.ray_step_cum[1])
	);

	n_rays_in_batch = n_batch_rays;
	n_samples_in_batch = n_batch_samples;
}

// update occupancy grid

void NeRFTrainingController::update_occupancy_grid(const cudaStream_t& stream, const float& selection_threshold) {
	const uint32_t grid_volume = nerf->occupancy_grid.volume_i;
	const uint32_t n_bitfield_bytes = nerf->occupancy_grid.get_n_bitfield_elements();
	const uint32_t n_levels = nerf->occupancy_grid.n_levels;
	const float inv_aabb_size = 1.0f / nerf->bounding_box.size_x;
	
	// decay occupancy grid values by 0.95
	decay_occupancy_grid_values_kernel<<<n_blocks_linear(grid_volume), n_threads_linear, 0, stream>>>(
		grid_volume,
		nerf->occupancy_grid.n_levels,
		0.95f,
		nerf->occupancy_grid.get_density()
	);

	// loop through each grid level, querying the network for the density at each cell and updating the occupancy grid's density
	for (int level = 0; level < n_levels; ++level) {

		// update occupancy grid values
		uint32_t n_cells_updated = 0;
		while (n_cells_updated < grid_volume) {
			uint32_t n_cells_to_update = std::min(grid_volume - n_cells_updated, workspace.batch_size);

			uint32_t batch_size = tcnn::next_multiple(n_cells_to_update, tcnn::batch_size_granularity);
			
			// generate random floats for sampling
			curandStatus_t status = curandGenerateUniform(rng, workspace.random_float, 4 * batch_size);
			if (status != CURAND_STATUS_SUCCESS) {
				printf("Error generating random floats for occupancy grid update.\n");
			}

			// generate random sampling points
			generate_grid_cell_network_sample_points_kernel<<<n_blocks_linear(n_cells_to_update), n_threads_linear, 0, stream>>>(
				n_cells_to_update,
				batch_size,
				n_cells_updated,
				workspace.occ_grid,
				level,
				inv_aabb_size,
				workspace.random_float,
				workspace.sample_pos[0]
			);

			// query the density network
			nerf->network.inference(
				stream,
				batch_size,
				workspace.sample_pos[0],
				nullptr,
				workspace.network_concat,
				workspace.network_output,
				false
			);

			// update occupancy grid values
			update_occupancy_with_density_kernel<<<n_blocks_linear(n_cells_to_update), n_threads_linear, 0, stream>>>(
				n_cells_to_update,
				n_cells_updated,
				level,
				selection_threshold,
				workspace.random_float + 3 * batch_size, // (random_float + 3 * batch_size) is so thresholding doesn't correspond to x,y,z positions
				workspace.network_output + 3 * batch_size,
				workspace.occ_grid
			);

			n_cells_updated += n_cells_to_update;
		}
	}

	// update the bits by thresholding the density values

	// This is adapted from the instant-NGP paper.  See page 15 on "Updating occupancy grids"
	const float threshold = 0.01f;// * NeRFConstants::min_step_size;

	update_occupancy_grid_bits_kernel<<<n_blocks_linear(n_bitfield_bytes), n_threads_linear, 0, stream>>>(
		nerf->occupancy_grid.volume_i,
		n_levels,
		threshold,
		workspace.occ_grid,
		nerf->occupancy_grid.get_density(),
		nerf->occupancy_grid.get_bitfield()
	);

	CHECK_DATA(bitfield_cpu, uint8_t, nerf->occupancy_grid.get_bitfield(), n_bitfield_bytes);

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
}

void NeRFTrainingController::train_step(const cudaStream_t& stream) {
	// Generate training batch
	generate_next_training_batch(stream);
	nerf->network.train(
		stream,
		workspace.batch_size,
		n_rays_in_batch,
		n_samples_in_batch,
		workspace.ray_step[1],
		workspace.ray_step_cum[1],
		workspace.sample_pos[1],
		workspace.sample_dir,
		workspace.sample_dt[1],
		workspace.pix_rgba[1],
		workspace.network_concat,
		workspace.network_output
	);

	++training_step;

}
