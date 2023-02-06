# pragma once
#include <cuda_runtime.h>
#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <vector>

#include "../common.h"
#include "../core/nerf-network.cuh"
#include "../core/occupancy-grid.cuh"
#include "../core/trainer.cuh"
#include "../models/bounding-box.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../workspaces/training-workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFTrainingController {
	// constructor
	NeRFTrainingController(Dataset& dataset, NeRFProxy* nerf_proxy, const uint32_t& batch_size);

	// public properties

	// public methods
	void prepare_for_training();
	void train_step();
	void update_occupancy_grid(const float& selection_threshold);

	uint32_t get_training_step() const {
		return training_step;
	}

private:
	// private properties
	std::vector<Trainer::Context> contexts;

	Dataset dataset;

	Trainer trainer;

	uint32_t training_step;
	
	// private methods
	void load_images(const cudaStream_t& stream, TrainingWorkspace& workspace);
};

NRC_NAMESPACE_END
