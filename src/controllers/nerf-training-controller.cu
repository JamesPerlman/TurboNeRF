
#include <device_launch_parameters.h>
#include <json/json.hpp>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/encoding.h>

#include <memory>
#include "nerf-training-controller.h"

using namespace nrc;
using namespace tcnn;
using namespace nlohmann;


NeRFTrainingController::NeRFTrainingController(
	const Dataset& dataset,
	const uint32_t& num_layers,
	const uint32_t& hidden_dim,
	const uint32_t& geo_feat_dim,
	const uint32_t& num_layers_color,
	const uint32_t& hidden_dim_color
) {
	this->dataset = dataset;
	this->num_layers = num_layers;
	this->hidden_dim = hidden_dim;
	this->geo_feat_dim = geo_feat_dim;
	this->num_layers_color = num_layers_color;
	this->hidden_dim_color = hidden_dim_color;
	
	// TODO: set this properly based on the aabb
	double per_level_scale = 1.4472692012786865;

	// Create the Direction Encoding
	
	json direction_encoding_config = {
		{"otype", "SphericalHarmonics"},
		{"degree", 4},
	};
	
	direction_encoding = std::shared_ptr<Encoding<network_precision_t>>(
		tcnn::create_encoding<network_precision_t>((uint32_t)3, direction_encoding_config)
	);
	
	// Create the Density MLP
	
	json density_mlp_encoding_config = {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", per_level_scale},
	};
	
	json density_mlp_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", hidden_dim},
		{"n_hidden_layers", num_layers - 1},
	};

	density_mlp = std::shared_ptr<tcnn::cpp::Module>(
		tcnn::cpp::create_network_with_input_encoding(3, 1 + geo_feat_dim, density_mlp_encoding_config, density_mlp_network_config)
	);
	
	// Create the Color MLP
	
	uint32_t color_mlp_in_dim = direction_encoding->padded_output_width() + geo_feat_dim;
		
	json color_mlp_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "Sigmoid"},
		{"n_neurons", hidden_dim_color},
		{"n_hidden_layers", num_layers_color - 1},
	};

	
	color_mlp = std::shared_ptr<tcnn::cpp::Module>(
		tcnn::cpp::create_network(color_mlp_in_dim, 3, color_mlp_network_config)
	);
	
	// Set up Optimizer
	
	json optimizer_config = {
		{"otype", "Adam"},
		{"learning_rate", 1e-2},
		{"epsilon", 1e-15},
	};
	
	optimizer = std::shared_ptr<Optimizer<tcnn::network_precision_t>>(
		create_optimizer<network_precision_t>(optimizer_config)
	);
	
	// RNG
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}

NeRFTrainingController::~NeRFTrainingController() {
	curandDestroyGenerator(rng);
}

__global__ void resizeFloatsToUInt32(uint32_t n_elements, float* floats, uint32_t* uints, float range_max) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n_elements) {
		float resized_val = floats[idx] * range_max;
		uints[idx] = (uint32_t)resized_val;
	}
}


void NeRFTrainingController::generate_next_training_batch(cudaStream_t stream, uint32_t training_step, uint32_t batch_size) {
	workspace.enlarge(stream, dataset.n_pixels_per_image, dataset.images.size(), batch_size);
	
	vector<uint32_t> random_indices_host(workspace.batch_size);
	// next, pull rays from the dataset 
	curandGenerateUniform(rng, workspace.random_floats, workspace.batch_size);
	resizeFloatsToUInt32<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(workspace.batch_size, workspace.random_floats, workspace.random_indices, dataset.n_pixels_per_image);

	cudaMemcpyAsync(random_indices_host.data(), workspace.random_indices, workspace.batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	
	printf("%ld", random_indices_host.size());
}

void NeRFTrainingController::train_step(cudaStream_t stream) {
	
	// Train the model (batch_size must be a multiple of tcnn::batch_size_granularity)
	uint32_t batch_size = tcnn::next_multiple((uint32_t)1000, tcnn::batch_size_granularity);
	
	generate_next_training_batch(stream, 1, batch_size);
	
	/*
	GPUMatrix<float> network_input(workspace.network_input);
	GPUMatrix<float> network_output(workspace.network_output);

	for (int i = 0; i < n_training_steps; ++i) {
		generate_training_batch(&training_batch_inputs, &training_batch_targets); // <-- your code

		float loss;
		model.trainer->training_step(training_stream, training_batch_inputs, training_batch_targets);
		std::cout << "iteration=" << i << " loss=" << loss << std::endl;
	}

	// Use the model
	GPUMatrix<float> inference_inputs(n_input_dims, batch_size);
	generate_inputs(&inference_inputs); // <-- your code

	GPUMatrix<float> inference_outputs(n_output_dims, batch_size);
	model.network->inference(inference_inputs, inference_outputs);
	*/
}