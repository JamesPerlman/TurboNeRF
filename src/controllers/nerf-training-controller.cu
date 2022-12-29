
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <json/json.hpp>
#include <Eigen/Dense>
#include <memory>
#include <thrust/device_vector.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

#include "nerf-training-controller.h"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../utils/training-batch-kernels.cuh"

using namespace nrc;
using namespace Eigen;
using namespace tcnn;
using namespace nlohmann;

NeRFTrainingController::NeRFTrainingController(Dataset& dataset) {
	this->dataset = dataset;
	
	network = NerfNetwork();

	// RNG
	// todo: CURAND_ASSERT_SUCCESS
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandGenerateSeeds(rng);

}

NeRFTrainingController::~NeRFTrainingController() {
	curandDestroyGenerator(rng);
}

// NeRFTrainingController member functions

void NeRFTrainingController::prepare_for_training(cudaStream_t stream, uint32_t batch_size) {
	// This allocates memory for all the elements we need during training
	workspace.enlarge(stream,
		dataset.n_pixels_per_image,
		dataset.n_channels_per_image,
		dataset.images.size(),
		batch_size,
		n_occupancy_grid_levels,
		occupancy_grid_resolution
	);

	// Initialize occupancy grid bitfield (all bits set to 1)
	CUDA_CHECK_THROW(
		cudaMemsetAsync(
			workspace.occupancy_grid_bitfield,
			(uint8_t)0b11111111, // set all bits to 1
			workspace.n_occupancy_grid_elements / 8,
			stream
		)
	);

	// Create a CascadedOccupancyGrid object and copy it to the GPU
	CascadedOccupancyGrid occupancy_grid_tmp(n_occupancy_grid_levels, workspace.occupancy_grid_bitfield, occupancy_grid_resolution);
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(workspace.occupancy_grid, &occupancy_grid_tmp, sizeof(CascadedOccupancyGrid), cudaMemcpyHostToDevice, stream)
	);

	// Copy dataset's BoundingBox to the GPU
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(workspace.bounding_box, &dataset.bounding_box, sizeof(BoundingBox), cudaMemcpyHostToDevice, stream)
	);

	// Training image indices will be reused for each batch.  We select the same number of rays from each image in the dataset.
	generate_training_image_indices<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		dataset.images.size(),
		workspace.img_index
	);
	
	// Copy training cameras to the GPU
	workspace.cameras.resize_and_copy_from_host(dataset.cameras);

	// Load all images into GPU memory!
	load_images(stream);
}

void NeRFTrainingController::load_images(cudaStream_t stream) {
	// make sure images are all loaded into CPU and GPU
	uint32_t image_size = dataset.n_channels_per_image * dataset.n_pixels_per_image * sizeof(stbi_uc);
	dataset.load_images_in_parallel(
		[this, &image_size, &stream](const size_t& image_index, const TrainingImage& image) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(
				workspace.image_data,
				image.data_cpu.get(),
				image_size,
				cudaMemcpyHostToDevice,
				stream
			));
		}
	);

	printf("All images loaded to GPU.\n");
}

#define CHECK_DATA(varname, data_type, data_ptr, data_size) \
	std::vector<data_type> varname(data_size); \
	CUDA_CHECK_THROW(cudaMemcpyAsync(varname.data(), data_ptr, data_size * sizeof(data_type), cudaMemcpyDeviceToHost, stream)); \
	cudaStreamSynchronize(stream);

/**
 * Based on my understanding of the instant-ngp paper and some help from NerfAcc,
  * we must do the following to generate a batch of fixed number of samples with a dynamic number of rays
  * 
  * 0. Generate rays and pixels
  * 1. Count the number of steps each ray will take
  * 2. Determine the maximum number of rays that will fill the batch with samples
  * 3. Generate the samples (t0, t1)
  * 4. Apply stratified sampling to get an array of t-values
  * 5. Run the network forward and get the predicted color and alpha for each sample
  * 6. Accumulate the colors and alphas from the color network output, along each ray
  * 7. Calculate the loss and backpropagate
 */

void NeRFTrainingController::generate_next_training_batch(cudaStream_t stream) {
	
	// Generate random floats for use in training
	
	curandStatus_t status = curandGenerateUniform(rng, workspace.random_floats, workspace.batch_size);
	if (status != CURAND_STATUS_SUCCESS) {
		printf("Error generating random floats for training batch.\n");
	}
	
	// Convert floats to uint32_t which will be interpreted as pixel indices for any training image
	resize_floats_to_uint32_with_max<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size, workspace.random_floats, workspace.pix_index, dataset.n_pixels_per_image
	);

	// Populate pixel buffers and ray data buffers based on the random numbers we generated
	initialize_training_rays_and_pixels_kernel<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		dataset.images.size(),
		dataset.n_pixels_per_image * dataset.n_channels_per_image,
		dataset.image_dimensions,
		workspace.cameras.data(),
		workspace.image_data,
		workspace.img_index,
		workspace.pix_index,
		workspace.pix_rgba[0],
		workspace.ori_xyz[0],
		workspace.dir_xyz[0],
		workspace.idir_xyz
	);

	CHECK_DATA(dirx1, float, workspace.dir_xyz[0] + 0 * workspace.batch_size, workspace.batch_size);
	CHECK_DATA(diry1, float, workspace.dir_xyz[0] + 1 * workspace.batch_size, workspace.batch_size);
	CHECK_DATA(dirz1, float, workspace.dir_xyz[0] + 2 * workspace.batch_size, workspace.batch_size);

	/* Begin volumetric sampling of the previous network outputs */
	
	// TODO: calculate these accurately

	float dt_min = 0.01f;
	float dt_max = 1.0f;
	float cone_angle = 1.0f;

	// Count the number of steps each ray would take
	march_and_count_steps_per_ray_kernel<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		workspace.bounding_box,
		workspace.occupancy_grid,
		cone_angle,
		dt_min,
		dt_max,
		workspace.ori_xyz[0],
		workspace.dir_xyz[0],
		workspace.idir_xyz,
		workspace.n_steps[0]
	);

	CHECK_DATA(nsteps1, uint32_t, workspace.n_steps[0], workspace.batch_size);

	// Grab some references to the double-buffered n_steps array
	thrust::device_ptr<uint32_t> n_steps_in_ptr(workspace.n_steps[0]);
	thrust::device_ptr<uint32_t> n_steps_cum_ptr(workspace.n_steps[1]);

	// Cumulative summation via inclusive_scan gives us the offset index that each ray's first sample should start at, relative to the start of the batch
	thrust::inclusive_scan(thrust::cuda::par.on(stream), n_steps_in_ptr, n_steps_in_ptr + workspace.batch_size, n_steps_cum_ptr);

	CHECK_DATA(nsteps2, uint32_t, workspace.n_steps[1], workspace.batch_size);

	// Populate the t0 and t1 buffers with the starts and ends of each ray's samples.
	// Also copy and compact other output buffers to help with coalesced memory access in future kernels.
	march_and_generate_samples_and_compact_buffers_kernel<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		workspace.bounding_box,
		workspace.occupancy_grid,
		dt_min, dt_max,
		cone_angle,
		
		// input buffers
		workspace.pix_rgba[0],
		workspace.ori_xyz[0],
		workspace.dir_xyz[0],
		workspace.idir_xyz,
		workspace.n_steps[0],
		workspace.n_steps[1],

		// output buffers
		workspace.pix_rgba[1],
		workspace.ori_xyz[1],
		workspace.dir_xyz[1],
		workspace.ray_t0, workspace.ray_t1
	);

	CHECK_DATA(rayt01, float, workspace.ray_t0, workspace.batch_size);
	CHECK_DATA(rayt11, float, workspace.ray_t1, workspace.batch_size);

	// Generate stratified sampling positions
	generate_stratified_sample_positions_kernel<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		workspace.ray_t0, workspace.ray_t1,
		workspace.random_floats,
		workspace.ori_xyz[1],
		workspace.dir_xyz[1],
		workspace.pos_xyz
	);

	// if this works im amazing
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void NeRFTrainingController::train_step(cudaStream_t stream) {
	
	// Train the model (batch_size must be a multiple of tcnn::batch_size_granularity)
	uint32_t batch_size = tcnn::next_multiple((uint32_t)1000, tcnn::batch_size_granularity);
	generate_next_training_batch(stream);
	
	// network.train(stream, batch_size, workspace.rgb_batch, workspace.dir_batch);
	
}