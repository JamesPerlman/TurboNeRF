
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

using namespace tcnn;
using namespace nlohmann;

TURBO_NAMESPACE_BEGIN

NeRFTrainingController::NeRFTrainingController(NeRFProxy* nerf_proxy, const uint32_t batch_size)
	: nerf_proxy(nerf_proxy)
{
	contexts.reserve(DeviceManager::get_device_count());
	
	DeviceManager::foreach_device(
		[this, nerf_proxy, batch_size](const int& device_id, const cudaStream_t& stream) {
			NeRF* nerf = &nerf_proxy->nerfs[device_id];
			contexts.emplace_back(
				stream,
				TrainingWorkspace(device_id),
				&nerf_proxy->dataset,
				nerf,
				NerfNetwork(device_id, nerf->aabb_scale()),
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
		ctx.batch_size,
		ctx.nerf->occupancy_grid.n_levels,
		ctx.nerf->occupancy_grid.resolution_i,
		ctx.network.get_concat_buffer_width(),
		ctx.network.get_padded_output_width()
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

	// make sure dataset workspace is allocated
	ctx.nerf->dataset_ws.enlarge(
		ctx.stream,
		ctx.dataset->images.size(),
		ctx.dataset->image_dimensions
	);

	// Copy dataset's BoundingBox to the GPU
	CUDA_CHECK_THROW(
		cudaMemcpyAsync(
			ctx.nerf->dataset_ws.bounding_box,
			&ctx.dataset->bounding_box,
			sizeof(BoundingBox),
			cudaMemcpyHostToDevice,
			ctx.stream
		)
	);

	// Load all images into GPU memory!
	load_images(ctx);

	training_step = 0;

	// Initialize the network
	ctx.network.prepare_for_training(ctx.stream, ctx.nerf->params);
}

void NeRFTrainingController::load_images(Trainer::Context& ctx) {
	// make sure images are all loaded into CPU and GPU
	// TODO: can we read images from a stream and load them directly into GPU memory? Probably!
	size_t n_image_elements = 4 * ctx.dataset->n_pixels_per_image;
	size_t image_size = n_image_elements * sizeof(stbi_uc);

	ctx.dataset->load_images_in_parallel(
		[this, &image_size, &n_image_elements, &ctx](const size_t& image_index, const TrainingImage& image) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(
				ctx.nerf->dataset_ws.image_data + image_index * n_image_elements,
				image.data_cpu.get(),
				image_size,
				cudaMemcpyHostToDevice,
				ctx.stream
			));
		}
	);

	printf("All images loaded to GPU.\n");
}

NeRFTrainingController::TrainingMetrics NeRFTrainingController::train_step() {

	// TODO: multi-gpu training.  For now we just train on the first device.
	auto& ctx = contexts[0];

	nerf_proxy->update_dataset_if_necessary(ctx.stream);
	
	float loss = trainer.train_step(ctx);
	++training_step;

	NeRFTrainingController::TrainingMetrics info;
	
	info.loss = loss;
	info.step = training_step;
	info.n_rays = ctx.n_rays_in_batch;
	info.n_samples = ctx.n_samples_in_batch;

	return info;
}

NeRFTrainingController::OccupancyGridMetrics NeRFTrainingController::update_occupancy_grid(const uint32_t& training_step) {
	// TODO: multi-gpu training.  For now we just update the occupancy grid on the first device.
	auto& ctx = contexts[0];
	uint32_t n_bits_occupied = trainer.update_occupancy_grid(ctx, training_step);

	NeRFTrainingController::OccupancyGridMetrics info;
	
	info.n_occupied = n_bits_occupied;
	info.n_total = ctx.nerf->occupancy_grid.get_n_total_elements();

	return info;
}

std::vector<size_t> NeRFTrainingController::get_cuda_memory_allocated() const {

    int n_gpus = DeviceManager::get_device_count();
    std::vector<size_t> sizes(n_gpus);

    // one context per GPU
    int i = 0;
    for (const auto& ctx : contexts) {
        size_t total = 0;

        total += ctx.workspace.get_bytes_allocated();
		total += ctx.network.workspace.get_bytes_allocated();

        sizes[i++] = total;
    }

	return sizes;
}

TURBO_NAMESPACE_END
