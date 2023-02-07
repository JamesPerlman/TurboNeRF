
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

using namespace nrc;
using namespace tcnn;
using namespace nlohmann;

NeRFTrainingController::NeRFTrainingController(Dataset& dataset, NeRFProxy* nerf_proxy, const uint32_t batch_size)
	: dataset(dataset)
{
	contexts.reserve(DeviceManager::get_device_count());
	for (int i = 0; i < DeviceManager::get_device_count(); ++i) {
		
		contexts.emplace_back(
			DeviceManager::get_stream(i), 
			TrainingWorkspace(i), 
			&this->dataset,
			&nerf_proxy->nerfs[i],
			batch_size
			);
	}
}

// NeRFTrainingController member functions

void NeRFTrainingController::prepare_for_training() {

	// we only prepare the first NeRF (for the first device) - the rest we will copy data to
	auto& ctx = contexts[0];

	// This allocates memory for all the elements we need during training
	ctx.workspace.enlarge(
		ctx.stream,
		ctx.dataset->images.size(),
		ctx.dataset->n_pixels_per_image,
		ctx.dataset->n_channels_per_image,
		ctx.batch_size,
		ctx.nerf->occupancy_grid.n_levels,
		ctx.nerf->occupancy_grid.resolution_i,
		ctx.nerf->network.get_concat_buffer_width(),
		ctx.nerf->network.get_padded_output_width()
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
	ctx.workspace.cameras.resize_and_copy_from_host(dataset.cameras);

	// Load all images into GPU memory!
	load_images(ctx.stream, ctx.workspace);

	// create the undistort map for camera 0 - assumption: all cameras have identical dist_params params
	trainer.create_pixel_undistort_map(ctx);

	training_step = 0;

	// Initialize the network
	ctx.nerf->network.prepare_for_training(ctx.stream);
}

void NeRFTrainingController::load_images(const cudaStream_t& stream, TrainingWorkspace& workspace) {
	// make sure images are all loaded into CPU and GPU
	// TODO: can we read images from a stream and load them directly into GPU memory? Probably!
	size_t n_image_elements = dataset.n_channels_per_image * dataset.n_pixels_per_image;
	size_t image_size = n_image_elements * sizeof(stbi_uc);

	dataset.load_images_in_parallel(
		[this, &image_size, &n_image_elements, &workspace, &stream](const size_t& image_index, const TrainingImage& image) {
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


void NeRFTrainingController::train_step() {
	// TODO: multi-gpu training.  For now we just train on the first device.
	auto& ctx = contexts[0];
	trainer.train_step(ctx);
	++training_step;
}

void NeRFTrainingController::update_occupancy_grid(const float& selection_threshold) {
	// TODO: multi-gpu training.  For now we just update the occupancy grid on the first device.
	auto& ctx = contexts[0];
	trainer.update_occupancy_grid(ctx, selection_threshold);
}
