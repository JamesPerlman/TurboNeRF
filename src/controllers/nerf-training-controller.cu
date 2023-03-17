
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <json/json.hpp>
#include <thrust/device_vector.h>
#include <tiny-cuda-nn/common.h>

#include "../services/device-manager.cuh"
#include "../utils/nerf-constants.cuh"
#include "../common.h"
#include "nerf-training-controller.h"

using namespace turbo;
using namespace tcnn;
using namespace nlohmann;

NeRFTrainingController::NeRFTrainingController(Dataset* dataset, NeRFProxy* nerf_proxy, const uint32_t batch_size)
{
	contexts.reserve(DeviceManager::get_device_count());
	DeviceManager::foreach_device(
		[this, nerf_proxy, batch_size, dataset](const int& device_id, const cudaStream_t& stream) {
			NeRF* nerf = &nerf_proxy->nerfs[device_id];
			contexts.emplace_back(
				stream,
				TrainingWorkspace(device_id),
				dataset,
				nerf,
				NerfNetwork(device_id),
				batch_size
			);
		}
	);
}

void NeRFTrainingController::prepare_for_training() {

	// we only prepare the first NeRF (for the first device) - the rest we will copy data to
	auto& ctx = contexts[0];
	
	// TODO: we should not initialize an occupancy grid if one already exists (aka if we loaded the nerf from a file)
	ctx.nerf->occupancy_grid.initialize(ctx.stream, true);

	// Initialize occupancy grid bitfield (all bits set to 1)
	ctx.nerf->occupancy_grid.set_bitfield(ctx.stream, 0b11111111);
	
	// Density can be set to zero, but probably doesn't need to be set at all
	ctx.nerf->occupancy_grid.set_density(ctx.stream, 0);
	
	// This allocates memory for all the elements we need during training
	ctx.workspace.enlarge(
		ctx.stream,
		ctx.dataset->images.size(),
		ctx.dataset->n_pixels_per_image,
		ctx.dataset->n_channels_per_image,
		ctx.batch_size,
		ctx.nerf->occupancy_grid.n_levels,
		ctx.nerf->occupancy_grid.resolution_i,
		ctx.network.get_concat_buffer_width(),
		ctx.network.get_padded_output_width()
	);

	// Copy dataset's BoundingBox to the GPU
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(
			ctx.workspace.bounding_box,
			&ctx.dataset->bounding_box,
			sizeof(BoundingBox),
			cudaMemcpyHostToDevice,
			ctx.stream
		)
	);

	// Copy nerf's OccupancyGrid to the GPU
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(
			ctx.workspace.occupancy_grid,
			&ctx.nerf->occupancy_grid,
			sizeof(OccupancyGrid),
			cudaMemcpyHostToDevice,
			ctx.stream
		)
	);

	// Copy training cameras to the GPU
	ctx.workspace.cameras.resize_and_copy_from_host(ctx.dataset->cameras);

	// Load all images into GPU memory!
	load_images(ctx);

	// create the undistort map for camera 0 - assumption: all cameras have identical dist_params params
	trainer.create_pixel_undistort_map(ctx);

	training_step = 0;

	// Initialize the network
	ctx.network.prepare_for_training(ctx.stream, ctx.nerf->params);
}

void NeRFTrainingController::load_images(Trainer::Context& ctx) {
	// make sure images are all loaded into CPU and GPU
	// TODO: can we read images from a stream and load them directly into GPU memory? Probably!
	size_t n_image_elements = ctx.dataset->n_channels_per_image * ctx.dataset->n_pixels_per_image;
	size_t image_size = n_image_elements * sizeof(stbi_uc);

	ctx.dataset->load_images_in_parallel(
		[this, &image_size, &n_image_elements, &ctx](const size_t& image_index, const TrainingImage& image) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(
				ctx.workspace.image_data + image_index * n_image_elements,
				image.data_cpu.get(),
				image_size,
				cudaMemcpyHostToDevice,
				ctx.stream
			));
		}
	);

	printf("All images loaded to GPU.\n");
}


void NeRFTrainingController::train_step() {
	// TODO: multi-gpu training.  For now we just train on the first device.
	auto& ctx = contexts[0];
	trainer.train_step(ctx);
	++training_step;
}

void NeRFTrainingController::update_occupancy_grid(const uint32_t& training_step) {
	// TODO: multi-gpu training.  For now we just update the occupancy grid on the first device.
	auto& ctx = contexts[0];
	trainer.update_occupancy_grid(ctx, training_step);
}
