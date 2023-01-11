// This code was adapted from nerfstudio (Copyright 2022 The Nerfstudio Team. All rights reserved.)
// https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/instant_ngp_field.py
// Please see LICENSES/nerfstudio-project_nerfstudio.md for license details.

#include <json/json.hpp>
#include <math.h>
#include <tiny-cuda-nn/common.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "../utils/gpu-image.cuh"
#include "../utils/parallel-utils.cuh"
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

NerfNetwork::NerfNetwork(const float& aabb_size) {
	this->aabb_size = aabb_size;

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

	uint32_t color_network_in_dim = direction_encoding->padded_output_width() + density_network->padded_output_width();

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
}

// initialize params and gradients for the networks (I have no idea if this is correct)
void NerfNetwork::prepare_for_training(const cudaStream_t& stream) {

	size_t rng_seed = 72791;
	pcg32 rng(rng_seed);

	// concatenated network params and gradients
	uint32_t n_total_params = density_network->n_params() + color_network->n_params();

	params_fp.enlarge(n_total_params);
	params_hp.enlarge(n_total_params);

	gradients_hp.enlarge(n_total_params);
	gradients_hp.memset(0);

	// initialize params

	density_network->initialize_params(rng, params_fp.data());

	color_network->initialize_params(rng, params_fp.data() + density_network->n_params());

	// initialize_params only initializes full precision params, need to copy to half precision

	copy_and_cast<network_precision_t, float>(stream, n_total_params, params_hp.data(), params_fp.data());

	// assign params pointers

	density_network->set_params(
		params_hp.data(),
		params_hp.data(),
		gradients_hp.data()
	);

	color_network->set_params(
		params_hp.data() + density_network->n_params(),
		params_hp.data() + density_network->n_params(),
		gradients_hp.data() + density_network->n_params()
	);

	// initialize optimizers
	
	json optimizer_config = {
		{"otype", "Adam"},
		{"learning_rate", 1e-2},
		{"epsilon", 1e-15},
	};

	density_optimizer.reset(
		create_optimizer<network_precision_t>(optimizer_config)
	);

	density_optimizer->allocate(density_network);

	color_optimizer.reset(
		create_optimizer<network_precision_t>(optimizer_config)
	);
	
	color_optimizer->allocate(color_network);

	// flag for training enabled
	can_train = true;
}

void NerfNetwork::train(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const uint32_t& n_rays,
	const uint32_t& n_samples,
	uint32_t* ray_steps,
	uint32_t* ray_steps_cum,
	float* pos_batch,
	float* dir_batch,
	float* dt_batch,
	float* target_rgba
) {
	enlarge_workspace_if_needed(stream, batch_size);

	// Normalize input for neural network
	generate_normalized_network_input(stream, batch_size, pos_batch, dir_batch, dt_batch);

	// Forward
	auto fwd_ctx = forward(stream, batch_size, workspace.normal_pos_batch, workspace.normal_dir_batch);

	// Loss
	float mse_loss = calculate_loss(
		stream,
		batch_size,
		n_rays,
		n_samples,
		ray_steps,
		ray_steps_cum,
		workspace.normal_dt_batch,
		target_rgba
	);

	printf("Loss: %f\n", mse_loss);

	// Backward
	backward(stream, fwd_ctx, batch_size, workspace.normal_pos_batch, workspace.normal_dir_batch, target_rgba);

	// Optimizer
	optimizer_step(stream);
}

// Normalizes input and saves it to the correct buffers (thank you @buriedanimal)
void NerfNetwork::generate_normalized_network_input(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const float* pos_batch,
	const float* dir_batch,
	const float* dt_batch
) {
	// Normalize input for neural network
	normalize_network_input_kernel<<<n_blocks_linear(batch_size), n_threads_linear, 0, stream>>>(
		batch_size,
		1.0f / aabb_size,
		pos_batch,
		dir_batch,
		dt_batch,
		workspace.normal_pos_batch,
		workspace.normal_dir_batch,
		workspace.normal_dt_batch
	);
}

void NerfNetwork::inference(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	float* pos_batch,
	float* dir_batch,
	// density network output must have space available for (color_network->input_width() * batch_size) elements of type network_precision_t
	network_precision_t* sigma,
	// color network output must have space available for (color_network->padded_output_width() * batch_size) elements of type network_precision_t
	network_precision_t* color
) {
	enlarge_workspace_if_needed(stream, batch_size);

	// Normalize input
	generate_normalized_network_input(stream, batch_size, pos_batch, dir_batch);

	// Inference (density network)
	GPUMatrix density_network_input_matrix(
		workspace.normal_pos_batch,
		density_network->input_width(),
		batch_size
	);

	GPUMatrix density_network_output_matrix(
		sigma,
		density_network->padded_output_width(),
		batch_size
	);

	density_network->inference_mixed_precision(
		stream,
		density_network_input_matrix,
		density_network_output_matrix
	);

	// Inference (direction encoding)
	network_precision_t* direction_encoding_output = color + density_network->output_width() * batch_size;

	GPUMatrix direction_encoding_input_matrix(
		workspace.normal_dir_batch,
		direction_encoding->input_width(),
		batch_size
	);

	GPUMatrix direction_encoding_output_matrix(
		direction_encoding_output,
		direction_encoding->padded_output_width(),
		batch_size
	);

	direction_encoding->inference_mixed_precision(
		stream,
		direction_encoding_input_matrix,
		direction_encoding_output_matrix
	);

	// Inference (color network)
	GPUMatrix color_network_input_matrix(
		density_network_output_matrix.data(),
		color_network->input_width(),
		batch_size
	);

	GPUMatrix color_network_output_matrix(
		color,
		color_network->padded_output_width(),
		batch_size
	);

	color_network->inference_mixed_precision(
		stream,
		color_network_input_matrix,
		color_network_output_matrix
	);
}

std::unique_ptr<NerfNetwork::ForwardContext> NerfNetwork::forward(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	float* pos_batch,
	float* dir_batch
) {
	auto fwd_ctx = std::make_unique<ForwardContext>();

	// Forward pass on density network (with multiresolution hash encoding built in!)

	fwd_ctx->density_network_input_matrix = GPUMatrix<float>(
		pos_batch,								// density network takes the sample positions as input
		density_network->input_width(),			// rows
		batch_size								// cols
	);

	// Here we make the output of the density network a pointer to the first half of the color network's input buffer.
	fwd_ctx->density_network_output_matrix = GPUMatrix<network_precision_t>(
		workspace.color_network_input, 			// density network output = color network input
		density_network->output_width(), 		// rows
		batch_size								// cols
	);

	fwd_ctx->density_ctx = density_network->forward(
		stream,
		fwd_ctx->density_network_input_matrix,
		&fwd_ctx->density_network_output_matrix,
		false,
		true // prepare_input_gradients must be `true` otherwise backwards() fails (forward->dy_dx is not defined)
	);

	// Encode directions (dir_batch)
	// Direction encoding gets concatenated with density_network_output (which will just be the second half of color_network_input)
	
	network_precision_t* direction_encoding_output = workspace.color_network_input + density_network->output_width() * batch_size;

	fwd_ctx->direction_encoding_input_matrix = GPUMatrix<float>(
		dir_batch,									// pointer to source data
		direction_encoding->input_width(),			// rows
		batch_size									// cols
	);

	fwd_ctx->direction_encoding_output_matrix = GPUMatrix<network_precision_t>(
		direction_encoding_output,					// pointer to destination data
		direction_encoding->padded_output_width(),	// rows
		batch_size									// cols
	);

	direction_encoding->inference_mixed_precision(
		stream,
		fwd_ctx->direction_encoding_input_matrix,
		fwd_ctx->direction_encoding_output_matrix
	);

	// Perform the forward pass on the color network

	fwd_ctx->color_network_input_matrix = GPUMatrix<network_precision_t>(
		workspace.color_network_input,				// pointer to source data
		color_network->input_width(),			// matrix rows
		batch_size								// matrix columns
	);

	fwd_ctx->color_network_output_matrix = GPUMatrix<network_precision_t>(
		workspace.color_network_output,			// pointer to destination data
		color_network->padded_output_width(),	// matrix rows
		batch_size								// matrix columns
	);

	fwd_ctx->color_ctx = color_network->forward(
		stream,
		fwd_ctx->color_network_input_matrix,
		&fwd_ctx->color_network_output_matrix,
		false,
		true // prepare_input_gradients
	);

	return fwd_ctx;
}

float NerfNetwork::calculate_loss(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const uint32_t& n_rays,
	const uint32_t& n_samples,
	const uint32_t* ray_steps,
	const uint32_t* ray_steps_cum,
	const float* sample_dt,
	const float* target_rgba
) {

	float n_raysf = n_rays;

	/**
	 * The density MLP maps the hash encoded position y = enc(x; 𝜃)
	 * to 16 output values, the first of which we treat as log-space density
	 * https://arxiv.org/abs/2201.05989 - Muller, et al. page 9
	 * 
	 */
	const tcnn::network_precision_t* log_space_density = workspace.color_network_input;

	accumulate_ray_colors_from_samples_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		ray_steps,
		ray_steps_cum,
		workspace.color_network_output,
		log_space_density,
		sample_dt,
		workspace.ray_rgba,
		workspace.trans_buf,
		workspace.alpha_buf,
		workspace.weight_buf
	);

	// Calculate mean-squared loss per ray
	calculate_sse_loss_per_ray_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		ray_steps,
		ray_steps_cum,
		workspace.ray_rgba,
		target_rgba,
		workspace.loss_buf,
		workspace.pxdiff_buf
	);

	
	// Zero out the gradient buffer
	CUDA_CHECK_THROW(
		cudaMemsetAsync(
			workspace.grad_buf,
			0,
			sizeof(network_precision_t) * color_network->padded_output_width() * batch_size,
			stream
		)
	);
	
	// Calculate gradients
	calculate_network_output_gradient<<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		batch_size,
		1.0f / (2.0f * n_raysf),
		workspace.color_network_output,
		workspace.pxdiff_buf,
		sample_dt,
		workspace.trans_buf,
		workspace.alpha_buf,
		workspace.weight_buf,
		LOSS_SCALE,
		workspace.grad_buf
	);

	// Add all loss values together
	thrust::device_ptr<float> loss_buffer_ptr(workspace.loss_buf);

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
	std::unique_ptr<NerfNetwork::ForwardContext>& fwd_ctx,
	uint32_t batch_size,
	float* pos_batch,
	float* dir_batch,
	float* target_rgba
) {
	// Backpropagate through the color network
	GPUMatrix<network_precision_t> color_network_dL_doutput_matrix(
		workspace.grad_buf,
		color_network->padded_output_width(),
		batch_size
	);

	GPUMatrix<network_precision_t> color_network_dL_dinput_matrix(
		workspace.color_network_dL_dinput,
		color_network->input_width(),
		batch_size
	);

	color_network->backward(
		stream,
		*fwd_ctx->color_ctx,
		fwd_ctx->color_network_input_matrix,
		fwd_ctx->color_network_output_matrix,
		color_network_dL_doutput_matrix,
		&color_network_dL_dinput_matrix
	);

	// Backpropagate through the density network
	GPUMatrix<float> density_network_dL_dinput_matrix(
		workspace.density_network_dL_dinput,
		density_network->input_width(),
		batch_size
	);

	// Construct a dL_dinput matrix of the correct size
	// color_network_dL_dinput_matrix is too large since it is the concatenation of density's outputs and encoded directions

	GPUMatrix<network_precision_t> density_network_dL_doutput_matrix(
		color_network_dL_dinput_matrix.data(),
		density_network->padded_output_width(),
		batch_size
	);

	density_network->backward(
		stream,
		*fwd_ctx->density_ctx,
		fwd_ctx->density_network_input_matrix,
		fwd_ctx->density_network_output_matrix,
		density_network_dL_doutput_matrix,
		&density_network_dL_dinput_matrix
	);

}

void NerfNetwork::optimizer_step(const cudaStream_t& stream) {

	density_optimizer->step(
		stream,
		LOSS_SCALE,
		params_fp.data(),
		params_hp.data(),
		gradients_hp.data()
	);

	color_optimizer->step(
		stream,
		LOSS_SCALE,
		params_fp.data() + density_network->n_params(),
		params_hp.data() + density_network->n_params(),
		gradients_hp.data() + density_network->n_params()
	);
}

// Only enlarge buffers needed for inference
void NerfNetwork::enlarge_workspace_if_needed(const cudaStream_t& stream, const uint32_t& batch_size) {
	if (batch_size <= this->batch_size) {
		return;
	}

	workspace.enlarge(
		stream,
		batch_size,
		density_network->input_width(),
		density_network->padded_output_width(),
		direction_encoding->input_width(),
		direction_encoding->padded_output_width(),
		color_network->input_width(),
		color_network->padded_output_width(),
		can_train
	);

	this->batch_size = batch_size;
}