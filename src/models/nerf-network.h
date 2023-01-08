#pragma once

#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct NerfNetwork {
	std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> direction_encoding;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> density_network;
	std::shared_ptr<tcnn::Network<tcnn::network_precision_t>> color_network;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> density_optimizer;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> color_optimizer;
	
	NerfNetwork(const float& aabb_size);

	void initialize_params(const cudaStream_t& stream);

	void train_step(
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
	);
	
	/**
	 * The density MLP maps the hash encoded position y = enc(x; 𝜃)
	 * to 16 output values, the first of which we treat as log-space density
	 * https://arxiv.org/abs/2201.05989 - page 9
	 */
	tcnn::network_precision_t* get_log_space_density() const {
		// The output of the density network is just a pointer to the color network's input buffer.
		return color_network_input.data();
	}

	tcnn::network_precision_t* get_color_network_output() const {
		return color_network_output.data();
	}

	size_t get_color_network_padded_output_width() const {
		return color_network->padded_output_width();
	}

private:

	float aabb_size;

	// full-precision params buffers for both MLPs
	tcnn::GPUMemory<float> params_fp;
	tcnn::GPUMemory<tcnn::network_precision_t> params_hp;
	tcnn::GPUMemory<tcnn::network_precision_t> gradients_hp;

	tcnn::GPUMemory<tcnn::network_precision_t> color_network_input;
	tcnn::GPUMemory<tcnn::network_precision_t> color_network_output;

	// gradient calculation buffers
	tcnn::GPUMemory<float> ray_rgba;
	tcnn::GPUMemory<float> loss_buf;
	tcnn::GPUMemory<tcnn::network_precision_t> grad_buf;
	tcnn::GPUMemory<float> trans_buf;
	tcnn::GPUMemory<float> alpha_buf;
	tcnn::GPUMemory<float> weight_buf; // alpha * transmittance
	tcnn::GPUMemory<float> pxdiff_buf; // pixel channel differences

	// buffers for normalized data
	tcnn::GPUMemory<float> normal_pos_batch;
	tcnn::GPUMemory<float> normal_dir_batch;
	tcnn::GPUMemory<float> normal_dt_batch;

	// Helper context
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrix<float> density_network_input_matrix;
		tcnn::GPUMatrix<tcnn::network_precision_t> density_network_output_matrix;

		tcnn::GPUMatrix<float> direction_encoding_input_matrix;
		tcnn::GPUMatrix<tcnn::network_precision_t> direction_encoding_output_matrix;
		
		tcnn::GPUMatrix<tcnn::network_precision_t> color_network_input_matrix;
		tcnn::GPUMatrix<tcnn::network_precision_t> color_network_output_matrix;
		
		tcnn::GPUMatrix<tcnn::network_precision_t> density_dL_doutput;
		tcnn::GPUMatrix<tcnn::network_precision_t> color_dL_doutput;
		tcnn::GPUMatrix<float> L;
		std::unique_ptr<tcnn::Context> density_ctx;
		std::unique_ptr<tcnn::Context> color_ctx;
	};

	void enlarge_batch_memory_if_needed(const uint32_t& batch_size);

	void generate_normalized_network_input(
		const cudaStream_t& stream,
		const uint32_t& batch_size,
		const float* pos_batch,
		const float* dir_batch,
		const float* dt_batch
	);

	std::unique_ptr<ForwardContext> forward(
		const cudaStream_t& stream,
		const uint32_t& batch_size,
		float* pos_batch,
		float* dir_batch
	);

	float calculate_loss(
		const cudaStream_t& stream,
		const uint32_t& batch_size,
		const uint32_t& n_rays,
		const uint32_t& n_samples,
		const uint32_t* ray_steps,
		const uint32_t* ray_steps_cumulative,
		const float* sample_dt,
		const float* target_rgba
	);

	void optimizer_step(const cudaStream_t& stream);

	void backward(
		cudaStream_t stream,
		std::unique_ptr<ForwardContext>& forward_ctx,
		uint32_t batch_size,
		float* pos_batch,
		float* dir_batch,
		float* target_rgba
	);
};

NRC_NAMESPACE_END
