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
#include "../models/bounding-box.h"
#include "../models/dataset.h"
#include "../models/training-workspace.h"

NRC_NAMESPACE_BEGIN

struct NeRFTrainingController {
	// constructor
	NeRFTrainingController(
		Dataset& dataset,
		const uint32_t& num_layers = 2,
		const uint32_t& hidden_dim = 64,
		const uint32_t& geo_feat_dim = 15,
		const uint32_t& num_layers_color = 3,
		const uint32_t& hidden_dim_color = 64
	);
	~NeRFTrainingController();

	// public methods
	void prepare_for_training(cudaStream_t stream, uint32_t batch_size);
	void load_images(cudaStream_t stream);
	void train_step(cudaStream_t stream);

private:
	// property members
	Dataset dataset;
	uint32_t num_layers;
	uint32_t hidden_dim;
	uint32_t geo_feat_dim;
	uint32_t num_layers_color;
	uint32_t hidden_dim_color;

	// network objects
	std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> direction_encoding;
	std::shared_ptr<tcnn::cpp::Module> density_mlp;
	std::shared_ptr<tcnn::cpp::Module> color_mlp;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;

	// workspace
	TrainingWorkspace workspace;
	curandGenerator_t rng;
	
	// private methods
	void generate_next_training_batch(cudaStream_t stream, uint32_t training_step);
};

NRC_NAMESPACE_END

