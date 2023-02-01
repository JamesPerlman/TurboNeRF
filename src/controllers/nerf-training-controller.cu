
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

	CHECK_DATA(ray_dir_cpu, float, workspace.ray_dir[0], n_rays_in_batch * 3);

	// Count the number of rays that will fill the batch with the maximum number of samples
	/**
	 * Cumulative summation via inclusive_scan gives us the offset index that each ray's first sample should start at, relative to the start of the batch.
	 * We need to perform this cumsum over the entire batch of rays, not just the rays that were regenerated over the used ones in the previous batch.
	 */

	// Grab some references to the n_steps arrays
	thrust::device_ptr<uint32_t> n_steps_ptr(workspace.ray_step[0]);
	thrust::device_ptr<uint32_t> ray_offset_ptr(workspace.ray_offset[0]);
	
	// cumulative sum the number of steps for each ray
	thrust::exclusive_scan(
		thrust::cuda::par.on(stream),
		n_steps_ptr,
		n_steps_ptr + workspace.batch_size,
		ray_offset_ptr
	);

	// Count the number of rays actually used to fill the sample batch
	const int n_ray_max_idx = find_last_lt_presorted(
		stream,
		ray_offset_ptr,
		workspace.batch_size,
		workspace.batch_size
	) - 1;

	if (n_rays_in_batch < 0) {
		throw std::runtime_error("No rays were generated for this training batch!\n");
	}

	n_rays_in_batch = static_cast<uint32_t>(n_ray_max_idx + 1);

	// Count the number of samples that will be generated
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(
			&n_samples_in_batch,
			ray_offset_ptr.get() + n_rays_in_batch,
			sizeof(uint32_t),
			cudaMemcpyDeviceToHost,
			stream
		)
	);

	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	if (n_samples_in_batch < 1) {
		throw std::runtime_error("No samples were generated for this training batch!\n");
	}
	
	// Generate sample positions
	march_and_generate_network_positions_kernel<<<n_blocks_linear(n_rays_in_batch), n_threads_linear, 0, stream>>>(
		n_rays_in_batch,
		workspace.batch_size,
		workspace.bounding_box,
		1.0f / dataset.bounding_box.size_x,
		workspace.occ_grid,
		dt_min,
		dt_max,
		cone_angle,

		// input buffers
		workspace.ray_origin,
		workspace.ray_dir[0],
		workspace.ray_inv_dir,
		workspace.ray_t,
		workspace.ray_offset[0],
		workspace.ray_alive,

		// dual-use buffers
		workspace.ray_step[0],

		// output buffers
		workspace.sample_pos[0],
		workspace.sample_dir,
		workspace.sample_dt[0]
	);
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
	// For some reason, the way the paper says it does not work for this implementation.
	// It seems to work with a threshold of 0.01
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
		workspace.ray_step[0],
		workspace.ray_offset[0],
		workspace.sample_pos[0],
		workspace.sample_dir,
		workspace.sample_dt[0],
		workspace.pix_rgba[0],
		workspace.network_concat,
		workspace.network_output
	);

	++training_step;

}
