# pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/optimizer.h>

#include "../common.h"
#include "../models/bounding-box.h"
#include "../models/dataset.h"
#include "../models/training-workspace.h"

NRC_NAMESPACE_BEGIN

struct NeRFTrainingController {
	NeRFTrainingController(
		const Dataset& dataset,
		const uint32_t& num_layers = 2,
		const uint32_t& hidden_dim = 64,
		const uint32_t& geo_feat_dim = 15,
		const uint32_t& num_layers_color = 3,
		const uint32_t& hidden_dim_color = 64
	);
	void train_step(cudaStream_t stream);
	
	~NeRFTrainingController();

private:
	// property members
	Dataset dataset;
	uint32_t num_layers;
	uint32_t hidden_dim;
	uint32_t geo_feat_dim;
	uint32_t num_layers_color;
	uint32_t hidden_dim_color;

	// network objects
	std::shared_ptr<tcnn::cpp::Module> direction_encoding;
	std::shared_ptr<tcnn::cpp::Module> density_mlp;
	std::shared_ptr<tcnn::cpp::Module> color_mlp;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;

	// workspace
	TrainingWorkspace workspace;
	curandGenerator_t rng;
	
	// private methods
	void generate_next_training_batch(cudaStream_t stream, uint32_t training_step, uint32_t batch_size);
};

NRC_NAMESPACE_END

