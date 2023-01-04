// This code was adapted from nerfstudio (Copyright 2022 The Nerfstudio Team. All rights reserved.)
// https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/instant_ngp_field.py
// Please see LICENSES/nerfstudio-project_nerfstudio.md for license details.

#include <json/json.hpp>
#include <tiny-cuda-nn/common.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "../utils/gpu_image.h"
#include "../utils/training-network-kernels.cuh"
#include "nerf-network.h"

using namespace tcnn;
using namespace nrc;
using json = nlohmann::json;


#if TCNN_HALF_PRECISION
    constexpr float LOSS_SCALE = 128.0f;
#else
    constexpr float LOSS_SCALE = 1.0f;
#endif


// Constructor

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

	// Set up Loss
	json loss_config = {
		{"otype", "L2"},
	};

	loss.reset(create_loss<network_precision_t>(loss_config));

	initialize_params_and_gradients();
}

// initialize params and gradients for the networks (I have no idea if this is correct)
void NerfNetwork::initialize_params_and_gradients() {

	size_t rng_seed = 72791;
	pcg32 rng(rng_seed);

	// concatenated network params and gradients
	uint32_t n_total_params = density_network->n_params() + color_network->n_params();

	params_fp.enlarge(n_total_params);
	params_fp.memset(0);

	params_hp.enlarge(n_total_params);
	params_hp.memset(0);

	gradients_hp.enlarge(n_total_params);
	gradients_hp.memset(0);

	// density network params and gradients

	density_network->initialize_params(rng, params_fp.data());
	density_network->set_params(
		params_hp.data(),
		params_hp.data(),
		gradients_hp.data()
	);

	// color network params and gradients

	color_network->initialize_params(rng, params_fp.data() + density_network->n_params());
	color_network->set_params(
		params_hp.data() + density_network->n_params(),
		params_hp.data() + density_network->n_params(),
		gradients_hp.data() + density_network->n_params()
	);

	// initialize optimizer
	
	std::vector<std::pair<uint32_t, uint32_t>> density_network_layer_sizes = density_network->layer_sizes();
	std::vector<std::pair<uint32_t, uint32_t>> color_network_layer_sizes = color_network->layer_sizes();

	std::vector<std::pair<uint32_t, uint32_t>> concatenated_layer_sizes(density_network_layer_sizes.size() + color_network_layer_sizes.size());
	concatenated_layer_sizes.insert(concatenated_layer_sizes.end(), density_network_layer_sizes.begin(), density_network_layer_sizes.end());
	concatenated_layer_sizes.insert(concatenated_layer_sizes.end(), color_network_layer_sizes.begin(), color_network_layer_sizes.end());

	optimizer->allocate(n_total_params, concatenated_layer_sizes);
}

std::unique_ptr<NerfNetwork::ForwardContext> NerfNetwork::forward(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	float* pos_batch,
	float* dir_batch
) {

	auto forward_ctx = std::make_unique<ForwardContext>();

	// Forward pass on density network (with multiresolution hash encoding built in!)

	GPUMatrix<float> density_network_input_matrix(
		pos_batch,								// density network takes the sample positions as input
		density_network->input_width(),			// rows
		batch_size								// cols
	);

	GPUMatrix<network_precision_t> density_network_output_matrix(
		color_network_input.data(), 			// density network outputs to the beginning of the color network input's data buffer
		density_network->output_width(), 		// rows
		batch_size								// cols
	);

	forward_ctx->density_ctx = density_network->forward(
		stream,
		density_network_input_matrix,
		&density_network_output_matrix,
		density_network->params(),
		true // TODO: research this (prepare_input_gradients) - do we need it to be `true` here?
	);

	// Encode directions (dir_batch)
	// Direction encoding gets concatenated with density_network_output (which will just be the end part of color_network_input)
	
	network_precision_t* direction_encoding_output = color_network_input.data() + density_network->output_width() * batch_size;

	GPUMatrix<float> direction_encoding_input_matrix(
		dir_batch,									// pointer to source data
		direction_encoding->input_width(),			// rows
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

	forward_ctx->color_ctx = color_network->forward(
		stream,
		color_network_input_matrix,
		&color_network_output_matrix,
		true,
		true
	);

	return forward_ctx;
}

float NerfNetwork::calculate_loss(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const uint32_t& n_rays,
	const uint32_t& n_samples,
	const uint32_t* ray_steps,
	const uint32_t* ray_steps_cumulative,
	const float* sample_dt,
	const float* target_rgba
) {

	float n_raysf = n_rays;

	/**
	 * The density MLP maps the hash encoded position y = enc(x; 𝜃)
	 * to 16 output values, the first of which we treat as log-space density
	 * https://arxiv.org/abs/2201.05989 - Muller, et al. page 9
	 * 
	 * i.e., the output of the density network is just a pointer to the color network's input buffer.
	 */
	const tcnn::network_precision_t* log_space_density = color_network_input.data();

	accumulate_ray_colors_from_samples_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		ray_steps,
		ray_steps_cumulative,
		color_network_output.data(),
		log_space_density,
		sample_dt,
		ray_rgba.data(),
		trans_buf.data(),
		alpha_buf.data(),
		weight_buf.data()
	);

	// Calculate mean-squared loss per ray
	calculate_ses_loss_per_ray_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		ray_steps,
		ray_steps_cumulative,
		ray_rgba.data(),
		target_rgba,
		loss_buf.data(),
		pxdiff_buf.data()
	);

	// Calculate gradients
	calculate_network_output_gradient<<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		batch_size,
		1.0f / (2.0f * n_raysf),
		color_network_output.data(),
		pxdiff_buf.data(),
		sample_dt,
		trans_buf.data(),
		alpha_buf.data(),
		weight_buf.data(),
		grad_buf.data()
	);

	// Add all loss values together
	thrust::device_ptr<float> loss_buffer_ptr(loss_buf.data());

	float sum_of_squared_pixel_errors = thrust::reduce(
		thrust::cuda::par_nosync.on(stream),
		loss_buffer_ptr,
		loss_buffer_ptr + n_rays,
		0.0f,
		thrust::plus<float>()
	);

	// Return mean loss
	return sum_of_squared_pixel_errors / (4.0f * n_raysf);
}

void NerfNetwork::backward(
	cudaStream_t stream,
	std::unique_ptr<NerfNetwork::ForwardContext>& forward_ctx,
	uint32_t batch_size,
	float* pos_batch,
	float* dir_batch,
	float* target_rgba
) {
	// ???

}

void NerfNetwork::train_step(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const uint32_t& n_rays,
	const uint32_t& n_samples,
	uint32_t* ray_steps,
	uint32_t* ray_steps_cumulative,
	float* pos_batch,
	float* dir_batch,
	float* dt_batch,
	float* target_rgba
) {
	enlarge_batch_memory_if_needed(batch_size);

	// Forward
	forward(stream, batch_size, pos_batch, dir_batch);

	// Loss
	float mse_loss = calculate_loss(
		stream,
		batch_size,
		n_rays,
		n_samples,
		ray_steps,
		ray_steps_cumulative,
		dt_batch,
		target_rgba
	);


	CHECK_DATA(lossdata_cpu, float, loss_buf.data(), batch_size);
	
	printf("%f\n", mse_loss);


	// Backward
	// backward(/* ??? */)

	// Optimizer
	// optimizer_step(stream, target_rgb, batch_size);
}

void NerfNetwork::optimizer_step(const cudaStream_t& stream) {

	optimizer->step(stream, LOSS_SCALE, params_fp.data(), params_hp.data(), gradients_hp.data());
/*
	GPUMatrix<network_precision_t> color_network_output_matrix(
		color_network_output.data(),
		color_network->padded_output_width(),
		batch_size,
		0
	);

	GPUMatrix<float> ground_truth_data_matrix(
		target_rgb,
		color_network->padded_output_width(),
		batch_size,
		0
	);
*/
/*
	// allocate matrices for dL_doutput and L
	GPUMatrix<network_precision_t> dL_doutput(color_network->padded_output_width(), batch_size, stream);
	GPUMatrix<float> L(color_network->padded_output_width(), batch_size, stream);
	
	loss->evaluate(stream,
		color_network->padded_output_width(),
		color_network->output_width(),
		128.0f, // no idea what loss_scale is supposed to be, it is 128.0f in trainer.h
		color_network_output_matrix,
		ground_truth_data_matrix,
		L,
		dL_doutput
		// last argument is data_pdf. Not sure if we need to use it. Default is nullptr.
	);
*/
}

void NerfNetwork::enlarge_batch_memory_if_needed(const uint32_t& batch_size) {
	uint32_t density_network_output_size = density_network->padded_output_width() * batch_size;
	uint32_t direction_encoding_output_size = direction_encoding->padded_output_width() * batch_size;
	color_network_input.enlarge(density_network_output_size + direction_encoding_output_size);
	color_network_output.enlarge(color_network->padded_output_width() * batch_size);

	ray_rgba.enlarge(4 * batch_size);
	loss_buf.enlarge(batch_size);
	grad_buf.enlarge(4 * batch_size);
	trans_buf.enlarge(batch_size);
	alpha_buf.enlarge(batch_size);
	weight_buf.enlarge(batch_size);
	pxdiff_buf.enlarge(4 * batch_size);
}
