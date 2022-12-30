// This code was adapted from nerfstudio (Copyright 2022 The Nerfstudio Team. All rights reserved.)
// https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/instant_ngp_field.py
// Please see LICENSES/nerfstudio-project_nerfstudio.md for license details.

#include <json/json.hpp>
#include <tiny-cuda-nn/common.h>
#include "../utils/gpu_image.h"
#include "nerf-network.h"

using namespace tcnn;
using namespace nrc;
using json = nlohmann::json;

NerfNetwork::NerfNetwork() {

	// TODO: set this properly based on the aabb
	double per_level_scale = 1.4472692012786865;

	// Create the Direction Encoding
	json direction_encoding_config = {
		{"otype", "SphericalHarmonics"},
		{"degree", 4},
	};

	direction_encoding.reset(
		create_encoding<network_precision_t>(3, direction_encoding_config)
	);

	// Create the Density MLP
	
	json density_encoding_config = {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", per_level_scale},
	};

	json density_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", 1},
	};

	density_network.reset(
		new NetworkWithInputEncoding<network_precision_t>(
			3,	// input dims
			16, // output dims
			density_encoding_config,
			density_network_config
		)
	);

	// Create the Color MLP

	uint32_t color_network_in_dim = direction_encoding->padded_output_width() + 16;

	const json color_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "Sigmoid"},
		{"n_neurons", 64},
		{"n_hidden_layers", 2},
		{"n_input_dims", color_network_in_dim},
		{"n_output_dims", 3},
	};

	color_network.reset(
		create_network<network_precision_t>(color_network_config)
	);

	// Set up Optimizer

	json optimizer_config = {
		{"otype", "Adam"},
		{"learning_rate", 1e-2},
		{"epsilon", 1e-15},
	};

	optimizer.reset(
		create_optimizer<network_precision_t>(optimizer_config)
	);

	initialize_params_and_gradients();
}

// initialize params and gradients for the networks
void NerfNetwork::initialize_params_and_gradients() {

	size_t rng_seed = 72791;
	pcg32 rng(rng_seed);

	// density network params and gradients

	density_network_params_fp.enlarge(density_network->n_params());
	density_network_params_fp.memset(0);

	density_network_params_hp.enlarge(density_network->n_params());
	density_network_params_hp.memset(0);

	density_network_gradients_hp.enlarge(density_network->n_params());
	density_network_gradients_hp.memset(0);
	
	density_network->initialize_params(rng, density_network_params_fp.data());
	density_network->set_params(density_network_params_hp.data(), density_network_params_hp.data(), density_network_gradients_hp.data());

	// color network params and gradients

	color_network_params_fp.enlarge(color_network->n_params());
	color_network_params_fp.memset(0);

	color_network_params_hp.enlarge(color_network->n_params());
	color_network_params_hp.memset(0);

	color_network_gradients_hp.enlarge(color_network->n_params());
	color_network_gradients_hp.memset(0);

	
	color_network->initialize_params(rng, color_network_params_fp.data());
	color_network->set_params(color_network_params_hp.data(), color_network_params_hp.data(), color_network_gradients_hp.data());
}

void NerfNetwork::forward(
	cudaStream_t stream,
	uint32_t batch_size,
	float* pos_batch,
	float* dir_batch
) {

	// Forward pass on density network (with multiresolution hash encoding built in!)

	GPUMatrix<float> density_network_input_matrix(
		pos_batch, 								// density network takes the sample positions as input
		density_network->input_width(), 		// rows
		batch_size 								// cols
	);

	GPUMatrix<network_precision_t> density_network_output_matrix(
		color_network_input.data(), 			// density network outputs to the beginning of the color network input's data buffer
		density_network->output_width(), 		// rows
		batch_size								// cols
	);

	density_network->forward(
		stream,
		density_network_input_matrix,
		&density_network_output_matrix,
		density_network_params_fp.data(),
		true // TODO: research this (prepare_input_gradients) - do we need it to be `true` here?
	);

	// Encode directions (dir_batch)
	// Direction encoding gets concatenated with density_network_output (which will just be the end part of color_network_input)
	
	network_precision_t* direction_encoding_output = color_network_input.data() + density_network->output_width() * batch_size;

	GPUMatrix<float> direction_encoding_input_matrix(
		dir_batch, 									// pointer to source data
		direction_encoding->input_width(), 			// rows
		batch_size									// cols
	);

	GPUMatrix<network_precision_t> direction_encoding_output_matrix(
		direction_encoding_output,					// pointer to destination data
		direction_encoding->padded_output_width(),	// rows
		batch_size									// cols
	);

	direction_encoding->inference_mixed_precision(stream, direction_encoding_input_matrix, direction_encoding_output_matrix);

	// Perform the forward pass on the color network

	GPUMatrix<network_precision_t> color_network_input_matrix(
		color_network_input.data(),				// pointer to source data
		color_network->input_width(),			// matrix rows
		batch_size,								// matrix columns
		0										// memory stride
	);

	GPUMatrix<network_precision_t> color_network_output_matrix(
		color_network_output.data(),			// pointer to destination data
		color_network->padded_output_width(),	// matrix rows
		batch_size,								// matrix columns
		0										// memory stride
	);
	
	color_network->forward(
		stream,
		color_network_input_matrix,
		&color_network_output_matrix,
		true,
		true
	);
}

void NerfNetwork::enlarge_batch_memory_if_needed(uint32_t batch_size) {
	uint32_t density_network_output_size = density_network->padded_output_width() * batch_size;
	uint32_t direction_encoding_output_size = direction_encoding->padded_output_width() * batch_size;
	color_network_input.enlarge(density_network_output_size + direction_encoding_output_size);
	color_network_output.enlarge(color_network->padded_output_width() * batch_size);
}
