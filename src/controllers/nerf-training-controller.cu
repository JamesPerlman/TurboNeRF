
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <json/json.hpp>
#include <Eigen/Dense>
#include <memory>
#include <thrust/device_vector.h>


#include "nerf-training-controller.h"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../utils/training-batch-kernels.cuh"
#include "../utils/parallel-utils.cuh"

using namespace nrc;
using namespace Eigen;
using namespace tcnn;
using namespace nlohmann;

NeRFTrainingController::NeRFTrainingController(Dataset& dataset)
	: network(NerfNetwork(dataset.bounding_box.size_x))
{
	this->dataset = dataset;
	
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
	network.initialize_params(stream);
}

void NeRFTrainingController::load_images(cudaStream_t stream) {
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
	resize_floats_to_uint32_with_max<<<n_blocks_linear(workspace.batch_size), n_threads_linear, 0, stream>>>(
		workspace.batch_size, workspace.random_floats, workspace.pix_index, dataset.n_pixels_per_image
	);

	/**
	 * Generate rays and pixels for training
	 * 
	 * We can take a shortcut here and generate only the data needed to fill the batch back up.
	 * If not all the previous batch's rays were used, then we can reuse the unused rays.
	 * AKA, n_rays_in_batch is the number of spent rays that need to be regenerated.
	 * 
	 * Huzzah, optimization!
	 * 
	 */
	initialize_training_rays_and_pixels_kernel<<<n_blocks_linear(n_rays_in_batch), n_threads_linear, 0, stream>>>(
		n_rays_in_batch,
		workspace.batch_size,
		dataset.images.size(),
		dataset.n_pixels_per_image * dataset.n_channels_per_image,
		dataset.image_dimensions,
		workspace.cameras.data(),
		workspace.image_data,
		workspace.img_index,
		workspace.pix_index,
		workspace.pix_rgba,
		workspace.ray_origins,
		workspace.ray_dirs,
		workspace.ray_inv_dirs
	);

	/* Begin volumetric sampling of the previous network outputs */
	
	// TODO: calculate these accurately

	float dt_min = 0.01f;
	float dt_max = 1.0f;
	float cone_angle = 1.0f;

	// Count the number of steps each ray would take.  We only need to do this for the new rays.
	march_and_count_steps_per_ray_kernel<<<n_blocks_linear(n_rays_in_batch), n_threads_linear, 0, stream>>>(
		n_rays_in_batch,
		workspace.batch_size,
		workspace.bounding_box,
		workspace.occupancy_grid,
		cone_angle,
		dt_min,
		dt_max,
		workspace.ray_origins,
		workspace.ray_dirs,
		workspace.ray_inv_dirs,
		workspace.ray_steps
	);

	/**
	 * Cumulative summation via inclusive_scan gives us the offset index that each ray's first sample should start at, relative to the start of the batch.
	 * We need to perform this cumsum over the entire batch of rays, not just the rays that were regenerated over the used ones in the previous batch.
	 */
	
	// Grab some references to the n_steps arrays
	thrust::device_ptr<uint32_t> n_steps_in_ptr(workspace.ray_steps);
	thrust::device_ptr<uint32_t> n_steps_cum_ptr(workspace.ray_steps_cumulative);

	// cumsum
	thrust::inclusive_scan(thrust::cuda::par.on(stream), n_steps_in_ptr, n_steps_in_ptr + workspace.batch_size, n_steps_cum_ptr);


	/**
	 * Populate the t0 and t1 buffers with the starts and ends of each ray's samples.
	 * Also copy and compact other output buffers to help with coalesced memory access in future kernels.
	 * Again, we perform this over the entire batch of samples.
	 */

	march_and_generate_samples_and_compact_buffers_kernel<<<n_blocks_linear(workspace.batch_size), n_threads_linear, 0, stream>>>(
		workspace.batch_size,
		workspace.bounding_box,
		workspace.occupancy_grid,
		dt_min, dt_max,
		cone_angle,
		
		// input buffers
		workspace.ray_origins,
		workspace.ray_dirs,
		workspace.ray_inv_dirs,
		workspace.ray_steps,
		workspace.ray_steps_cumulative,
		
		// output buffers
		workspace.sample_origins,
		workspace.sample_dirs,
		workspace.sample_t0,
		workspace.sample_t1
	);

	// Generate stratified sampling positions
	generate_stratified_sample_positions_kernel<<<n_blocks_linear(workspace.batch_size), n_threads_linear, 0, stream>>>(
		workspace.batch_size,
		workspace.sample_t0, workspace.sample_t1,
		workspace.random_floats,
		workspace.sample_origins,
		workspace.sample_dirs,
		workspace.sample_positions,
		workspace.sample_dt
	);

	// Count the number of rays actually used to fill the sample batch

	int n_ray_max_idx = find_last_lt_presorted(
		stream,
		n_steps_cum_ptr,
		workspace.batch_size,
		workspace.batch_size
	);

	if (n_ray_max_idx <= 0) {
		// TODO: better error handling
		throw std::runtime_error("Sample batch does not contain any rays!\n");
	}

	n_rays_in_batch = n_ray_max_idx + 1;
	cudaMemcpyAsync(&n_samples_in_batch, n_steps_cum_ptr.get() + n_ray_max_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
}

void NeRFTrainingController::train_step(cudaStream_t stream) {
	printf("Training step %d...\n", training_step);

	// Generate training batch
	generate_next_training_batch(stream);

	printf("Using %d rays and %d samples\n", n_rays_in_batch, n_samples_in_batch);
	
	// TODO: NORMALIZE SAMPLE_DIRS AND SAMPLE_POSITIONS (THANK YOU @BURIEDANIMAL)
	network.train_step(
		stream,
		workspace.batch_size,
		n_rays_in_batch,
		n_samples_in_batch,
		workspace.ray_steps,
		workspace.ray_steps_cumulative,
		workspace.sample_positions,
		workspace.sample_dirs,
		workspace.sample_dt,
		workspace.pix_rgba
	);

	++training_step;

}
