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
#include "nerf-network.cuh"

using namespace tcnn;
using namespace nrc;
using json = nlohmann::json;


#if TCNN_HALF_PRECISION
    constexpr float LOSS_SCALE = 128.0f;
#else
    constexpr float LOSS_SCALE = 1.0f;
#endif


// Constructor

NerfNetwork::NerfNetwork(
	const int& device_id,
	const float& aabb_size
)
	: network_ws(device_id)
	, params_ws(device_id)
{
	this->aabb_size = aabb_size;

	// These values are from the Instant-NGP paper, page 4. "Multiresolution Hash Encoding"
	double n_levels = 16.0;
	double N_min = 16.0;
	double N_max = 524288.0;
	double b = exp((log(N_max) - log(N_min)) / (n_levels - 1.0));

	// These network configurations were adapted from nerfstudio
	
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
		{"per_level_scale", b},
		// used by recommendation of Müller et al (instant-NGP paper, page 13 "Smooth Interpolation")
		{"interpolation", "Smoothstep"},
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

	// initialize params
	params_ws.enlarge(
		stream,
		density_network->n_params(),
		color_network->n_params()
	);
	
	density_network->initialize_params(rng, params_ws.density_network_params_fp);
	color_network->initialize_params(rng, params_ws.color_network_params_fp);

	// initialize_params only initializes full precision params, need to copy to half precision

	copy_and_cast<network_precision_t, float>(
		stream,
		density_network->n_params(),
		params_ws.density_network_params_hp,
		params_ws.density_network_params_fp
	);

	copy_and_cast<network_precision_t, float>(
		stream,
		color_network->n_params(),
		params_ws.color_network_params_hp,
		params_ws.color_network_params_fp
	);

	// assign params pointers

	density_network->set_params(
		params_ws.density_network_params_hp,
		params_ws.density_network_params_hp,
		params_ws.density_network_gradients_hp
	);

	color_network->set_params(
		params_ws.color_network_params_hp,
		params_ws.color_network_params_hp,
		params_ws.color_network_gradients_hp
	);

	// initialize optimizers
	
	json optimizer_config = {
		{"otype", "Adam"},
		{"learning_rate", 1e-2},
		{"epsilon", 1e-15},
		{"l2_reg", 1e-6}
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
	uint32_t* ray_offset,
	float* pos_batch,
	float* dir_batch,
	float* dt_batch,
	float* target_rgba,
	network_precision_t* concat_buffer,
	network_precision_t* output_buffer
) {
	
	enlarge_workspace_if_needed(stream, batch_size);

	// Forward
	auto fwd_ctx = forward(
		stream,
		batch_size,
		n_rays,
		n_samples,
		ray_steps,
		ray_offset,
		target_rgba,
		pos_batch,
		dir_batch,
		dt_batch,
		concat_buffer,
		output_buffer
	);

	// Loss
	const float loss = calculate_loss(
		stream,
		batch_size,
		n_rays
	);

	printf(
		"Loss: %f / # Rays: %u\n",
		loss,
		n_rays
	);

	// Backward
	backward(
		stream,
		fwd_ctx,
		n_rays,
		n_samples,
		batch_size,
		ray_steps,
		ray_offset,
		concat_buffer,
		output_buffer,
		pos_batch,
		dir_batch,
		dt_batch,
		target_rgba
	);

	// Optimizer
	optimizer_step(stream);
}

void NerfNetwork::inference(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	float* pos_batch,
	float* dir_batch,
	// density network output must have space available for (color_network->input_width() * batch_size) elements of type network_precision_t
	network_precision_t* concat_buffer,
	// color network output must have space available for (color_network->padded_output_width() * batch_size) elements of type network_precision_t
	network_precision_t* output_buffer,
	// if this flag is false, we only run inference on the density network
	const bool& use_color_network
) {
	// Inference (density network)
	GPUMatrixDynamic density_network_input_matrix(
		pos_batch,
		density_network->input_width(),
		batch_size,
		MatrixLayout::RowMajor
	);

	GPUMatrixDynamic density_network_output_matrix(
		concat_buffer,
		density_network->padded_output_width(),
		batch_size,
		MatrixLayout::RowMajor
	);

	density_network->inference_mixed_precision(
		stream,
		density_network_input_matrix,
		density_network_output_matrix
	);

	if (use_color_network) {
		// Inference (direction encoding)
		network_precision_t* direction_encoding_output = concat_buffer + density_network->padded_output_width() * batch_size;

		GPUMatrixDynamic direction_encoding_input_matrix(
			dir_batch,
			direction_encoding->input_width(),
			batch_size,
			MatrixLayout::RowMajor
		);

		GPUMatrixDynamic direction_encoding_output_matrix(
			direction_encoding_output,
			direction_encoding->padded_output_width(),
			batch_size,
			MatrixLayout::RowMajor
		);

		direction_encoding->inference_mixed_precision(
			stream,
			direction_encoding_input_matrix,
			direction_encoding_output_matrix
		);

		// Inference (color network)
		GPUMatrixDynamic color_network_input_matrix(
			density_network_output_matrix.data(),
			color_network->input_width(),
			batch_size,
			MatrixLayout::RowMajor
		);

		GPUMatrixDynamic color_network_output_matrix(
			output_buffer,
			color_network->padded_output_width(),
			batch_size,
			MatrixLayout::RowMajor
		);

		color_network->inference_mixed_precision(
			stream,
			color_network_input_matrix,
			color_network_output_matrix
		);
	}

	// for inference we just overwrite the color network's alpha channel with activated sigma data
	density_to_sigma_forward_kernel<<<n_blocks_linear(batch_size), tcnn::n_threads_linear, 0, stream>>>(
		batch_size,
		concat_buffer,
		output_buffer + 3 * batch_size
	);
}

std::unique_ptr<NerfNetwork::ForwardContext> NerfNetwork::forward(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const uint32_t& n_rays,
	const uint32_t& n_samples,
	const uint32_t* ray_steps,
	const uint32_t* ray_offset,
	const float* target_rgba,
	float* pos_batch,
	float* dir_batch,
	float* dt_batch,
	network_precision_t* concat_buffer,
	network_precision_t* output_buffer
) {
	auto fwd_ctx = std::make_unique<ForwardContext>();

	// Forward pass on density network (with multiresolution hash encoding built in!)

	fwd_ctx->density_network_input_matrix = GPUMatrixDynamic(
		pos_batch,								// density network takes the sample positions as input
		density_network->input_width(),			// rows
		batch_size,								// cols
		MatrixLayout::RowMajor
	);

	// Here we make the output of the density network a pointer to the first half of the color network's input buffer.
	fwd_ctx->density_network_output_matrix = GPUMatrixDynamic(
		concat_buffer,				 			// density network output = color network input
		density_network->output_width(), 		// rows
		batch_size,								// cols
		MatrixLayout::RowMajor
	);

	fwd_ctx->density_ctx = density_network->forward(
		stream,
		fwd_ctx->density_network_input_matrix,
		&fwd_ctx->density_network_output_matrix,
		false,
		true // prepare_input_gradients must be `true` otherwise backward() fails (forward->dy_dx is not defined)
	);

	// Encode directions (dir_batch)
	// Direction encoding gets concatenated with density_network_output (which will just be the second half of concat_buffer)

	network_precision_t* direction_encoding_output = concat_buffer + density_network->padded_output_width() * batch_size;

	fwd_ctx->direction_encoding_input_matrix = GPUMatrixDynamic(
		dir_batch,									// pointer to source data
		direction_encoding->input_width(),			// rows
		batch_size,									// cols
		MatrixLayout::RowMajor
	);

	fwd_ctx->direction_encoding_output_matrix = GPUMatrixDynamic(
		direction_encoding_output,					// pointer to destination data
		direction_encoding->padded_output_width(),	// rows
		batch_size,									// cols
		MatrixLayout::RowMajor
	);

	direction_encoding->forward(
		stream,
		fwd_ctx->direction_encoding_input_matrix,
		&fwd_ctx->direction_encoding_output_matrix
	);

	// Perform the forward pass on the color network

	fwd_ctx->color_network_input_matrix = GPUMatrixDynamic(
		concat_buffer,							// pointer to source data
		color_network->input_width(),			// matrix rows
		batch_size,								// matrix columns
		MatrixLayout::RowMajor
	);

	fwd_ctx->color_network_output_matrix = GPUMatrixDynamic(
		output_buffer,							// pointer to destination data
		color_network->padded_output_width(),	// matrix rows
		batch_size,								// matrix columns
		MatrixLayout::RowMajor
	);

	fwd_ctx->color_ctx = color_network->forward(
		stream,
		fwd_ctx->color_network_input_matrix,
		&fwd_ctx->color_network_output_matrix,
		false,
		true // prepare_input_gradients
	);
	
	// Zero out transmittance
	cudaMemsetAsync(network_ws.trans_buf, 0, batch_size * sizeof(float), stream);

	// Continue forward with custom operators
	density_to_sigma_forward_kernel<<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		concat_buffer,
		network_ws.sigma_buf
	);

	sigma_to_alpha_forward_kernel<<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		network_ws.sigma_buf,
		dt_batch,
		network_ws.alpha_buf
	);

	sigma_to_ray_rgba_forward_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		ray_steps,
		ray_offset,
		network_ws.sigma_buf,
		output_buffer,
		network_ws.alpha_buf,
		network_ws.ray_rgba
	);

	ray_rgba_to_loss_forward_kernel<<<n_blocks_linear(batch_size), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		network_ws.ray_rgba,
		target_rgba,
		network_ws.loss_buf
	);

	return fwd_ctx;
}

float NerfNetwork::calculate_loss(
	const cudaStream_t& stream,
	const uint32_t& batch_size,
	const uint32_t& n_rays
) {
	// Add all loss values together
	thrust::device_ptr<float> loss_buffer_ptr(network_ws.loss_buf);

	return (1.0f / (float)n_rays) * thrust::reduce(
		thrust::cuda::par_nosync.on(stream),
		loss_buffer_ptr,
		loss_buffer_ptr + 4 * batch_size,
		0.0f,
		thrust::plus<float>()
	);
}

void NerfNetwork::backward(
	const cudaStream_t& stream,
	const std::unique_ptr<NerfNetwork::ForwardContext>& fwd_ctx,
	const uint32_t& n_rays,
	const uint32_t& n_samples,
	const uint32_t& batch_size,
	const uint32_t* ray_steps,
	const uint32_t* ray_offset,
	const tcnn::network_precision_t* network_density,
	const tcnn::network_precision_t* network_color,
	float* pos_batch,
	float* dir_batch,
	float* dt_batch,
	float* target_rgba
) {
	// zero out previous gradients
	cudaMemsetAsync(network_ws.grad_dL_dR, 0, 4 * batch_size * sizeof(float), stream);
	cudaMemsetAsync(network_ws.grad_dL_dcolor, 0, 3 * batch_size * sizeof(float), stream);
	cudaMemsetAsync(network_ws.grad_dL_dsigma, 0, batch_size * sizeof(float), stream);

	// Backpropagate loss
	ray_rgba_to_loss_backward_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		1.0f / (2.0f * (float)n_rays),
		network_ws.ray_rgba,
		target_rgba,
		network_ws.grad_dL_dR
	);

	sigma_to_ray_rgba_backward_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
		n_rays,
		batch_size,
		ray_steps,
		ray_offset,
		dt_batch,
		network_ws.alpha_buf,
		network_color,
		network_ws.ray_rgba,
		network_ws.grad_dL_dR,
		network_ws.grad_dL_dsigma,
		network_ws.grad_dL_dcolor
	);

	density_to_sigma_backward_kernel<<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		network_density,
		network_ws.grad_dL_dsigma,
		network_ws.grad_dL_ddensity
	);

	// Backpropagate through the color network
	GPUMatrixDynamic color_network_dL_doutput_matrix(
		network_ws.color_network_dL_doutput,
		color_network->padded_output_width(),
		batch_size,
		MatrixLayout::RowMajor
	);

	// need to clear & populate color network matrix
	color_network_dL_doutput_matrix.memset_async(stream, 0);

	copy_gradients_kernel<3, false><<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		batch_size,
		LOSS_SCALE,
		network_ws.grad_dL_dcolor,
		color_network_dL_doutput_matrix.data()
	);

	GPUMatrixDynamic color_network_dL_dinput_matrix(
		network_ws.color_network_dL_dinput,
		color_network->input_width(),
		batch_size,
		MatrixLayout::RowMajor
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
	GPUMatrixDynamic density_network_dL_dinput_matrix(
		network_ws.density_network_dL_dinput,
		density_network->input_width(),
		batch_size,
		MatrixLayout::RowMajor
	);

	// Construct a dL_dinput matrix of the correct size
	// color_network_dL_dinput_matrix is too large since it is the concatenation of density's outputs and encoded directions

	GPUMatrixDynamic density_network_dL_doutput_matrix(
		color_network_dL_dinput_matrix.data(),
		density_network->padded_output_width(),
		batch_size,
		MatrixLayout::RowMajor
	);

	// We need to add dL/ddensity to dL/doutput before backpropagating
	copy_gradients_kernel<1, true><<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
		n_samples,
		batch_size,
		LOSS_SCALE,
		network_ws.grad_dL_ddensity,
		density_network_dL_doutput_matrix.data()
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
		params_ws.density_network_params_fp,
		params_ws.density_network_params_hp,
		params_ws.density_network_gradients_hp
	);

	color_optimizer->step(
		stream,
		LOSS_SCALE,
		params_ws.color_network_params_fp,
		params_ws.color_network_params_hp,
		params_ws.color_network_gradients_hp
	);
}

// Only enlarge buffers needed for inference
void NerfNetwork::enlarge_workspace_if_needed(const cudaStream_t& stream, const uint32_t& batch_size) {
	if (batch_size <= this->batch_size) {
		return;
	}

	network_ws.enlarge(
		stream,
		batch_size,
		density_network->input_width(),
		density_network->padded_output_width(),
		direction_encoding->input_width(),
		direction_encoding->padded_output_width(),
		color_network->input_width(),
		color_network->padded_output_width()
	);

	this->batch_size = batch_size;
}
