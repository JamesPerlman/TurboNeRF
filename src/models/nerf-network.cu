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

	const uint32_t num_layers = 2;
	const uint32_t hidden_dim = 64;
	const uint32_t geo_feat_dim = 15;
	const uint32_t num_layers_color = 3;
	const uint32_t hidden_dim_color = 64;

	// TODO: set this properly based on the aabb
	double per_level_scale = 1.4472692012786865;

	// Create the Direction Encoding
	json direction_encoding_config = {
		{"otype", "SphericalHarmonics"},
		{"degree", 4},
	};

	direction_encoding = std::shared_ptr<Encoding<network_precision_t>>(
		tcnn::create_encoding<network_precision_t>(3, direction_encoding_config)
	);

	// Create the Density MLP
	
	json density_network_encoding_config = {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", per_level_scale},
	};

	json density_network_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", num_layers - 1},
	};

	density_network = std::shared_ptr<cpp::Module>(
		cpp::create_network_with_input_encoding(3, 16, density_network_encoding_config, density_network_network_config)
	);

	// Create the Color MLP

	uint32_t color_network_in_dim = direction_encoding->padded_output_width() + 16;

	json color_network_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "Sigmoid"},
		{"n_neurons", 64},
		{"n_hidden_layers", num_layers_color - 1},
	};

	color_network = std::shared_ptr<tcnn::cpp::Module>(
		cpp::create_network(color_network_in_dim, 3, color_network_network_config)
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

	// Allocate memory
	
	
	tcnn::pcg32 rng{ 72791 };
	density_network_params_fp.enlarge(density_network->n_params() * sizeof(float));
	density_network_params_fp.memset(0);
	
	//= GPUMemory<float>(density_network->n_params());
	density_network->initialize_params(72791, density_network_params_fp.data());
	
	//direction_encoding_params_full_precision.resize(3 * direction_encoding->n_params() * sizeof(float));
	//direction_encoding->initialize_params(rnd, direction_encoding_params_full_precision.data());
}

void NerfNetwork::train(cudaStream_t stream, uint32_t batch_size, float* rgb_batch, float* dir_batch) {
	enlarge_batch_memory_if_needed(batch_size);

	// TODO: research prepare_input_gradients - do we need it to be `true` here?
	network_precision_t* density_network_output = color_network_input.data();
	density_network->forward(stream, batch_size, rgb_batch, density_network_output, density_network_params_fp.data(), true);

	// Direction encoding gets concatenated with density_network_output.  To save time copying things, we use the buffer color_network_input.
	network_precision_t* direction_encoding_output = color_network_input.data() + density_network->n_output_dims() * batch_size;
	GPUMatrix<float> direction_encoding_input_matrix(dir_batch, 3, batch_size, 0);
	GPUMatrix<network_precision_t> direction_encoding_output_matrix(direction_encoding_output, direction_encoding->padded_output_width(), batch_size, 0);

	direction_encoding->inference_mixed_precision(stream, direction_encoding_input_matrix, direction_encoding_output_matrix);
	
	nrc::save_buffer_to_image("H:\\test.png", color_network_input.data(), batch_size, 32, 1);
	// nrc::save_buffer_to_image("H:\\testrgb.png", rgb_batch, batch_size, 3, 1);
	// nrc::save_buffer_to_image("H:\\testdir.png", dir_batch, batch_size, 3, 1);
}

void NerfNetwork::enlarge_batch_memory_if_needed(uint32_t batch_size) {
	uint32_t density_network_output_size = density_network->n_output_dims() * batch_size;
	uint32_t direction_encoding_output_size = direction_encoding->padded_output_width() * batch_size;
	color_network_input.enlarge(density_network_output_size + direction_encoding_output_size);
}
