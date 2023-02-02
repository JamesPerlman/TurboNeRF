# pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include "../common.h"
#include "../models/bounding-box.cuh"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../models/dataset.h"
#include "../models/nerf-network.h"
#include "../models/nerf.cuh"
#include "../models/training-workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFTrainingController {
	// constructor
	NeRFTrainingController(Dataset& dataset, NeRF* nerf);
	~NeRFTrainingController();
	
	// public properties

	// public methods
	void prepare_for_training(const cudaStream_t& stream, const uint32_t& batch_size);
	void load_images(const cudaStream_t& stream);
	void train_step(const cudaStream_t& stream);
	void update_occupancy_grid(const cudaStream_t& stream, const float& selection_threshold);

	uint32_t get_training_step() const {
		return training_step;
	}

private:
	// private properties
	NeRF* nerf;
	Dataset& dataset;

	uint32_t training_step;
	uint32_t n_rays_in_batch;
	uint32_t n_samples_in_batch;

	// workspace
	TrainingWorkspace workspace;
	curandGenerator_t rng;
	
	// private methods
	void generate_next_training_batch(const cudaStream_t& stream);
};

NRC_NAMESPACE_END

