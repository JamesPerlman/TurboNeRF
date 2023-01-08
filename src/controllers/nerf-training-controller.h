# pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
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
#include "../models/training-workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFTrainingController {
	// constructor
	NeRFTrainingController(Dataset& dataset);
	~NeRFTrainingController();
	
	// public properties

	// public methods
	void prepare_for_training(cudaStream_t stream, uint32_t batch_size);
	void load_images(cudaStream_t stream);
	void train_step(cudaStream_t stream);

	uint32_t get_training_step() const {
		return training_step;
	}

private:
	// private properties
	Dataset dataset;
	uint32_t training_step;
	uint32_t n_rays_in_batch;
	uint32_t n_samples_in_batch;

	// configuration properties
	uint32_t n_occupancy_grid_levels = 5;
	uint32_t occupancy_grid_resolution = 128;

	// network objects
	NerfNetwork network;

	// workspace
	TrainingWorkspace workspace;
	curandGenerator_t rng;
	
	// private methods
	void generate_next_training_batch(cudaStream_t stream);
};

NRC_NAMESPACE_END

