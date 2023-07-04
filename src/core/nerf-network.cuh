#pragma once

#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include "../core/adam-optimizer.cuh"
#include "../workspaces/network-workspace.cuh"
#include "../workspaces/network-params-workspace.cuh"
#include "../common.h"

TURBO_NAMESPACE_BEGIN

struct NerfNetwork {
	
	NetworkWorkspace workspace;

	std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> direction_encoding;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> density_network;
	std::shared_ptr<tcnn::Network<tcnn::network_precision_t>> color_network;
	std::shared_ptr<tcnn::NGPAdamOptimizer<tcnn::network_precision_t>> optimizer;
	
	NerfNetwork(const int& device_id);

	void update_params_if_needed(const cudaStream_t& stream, NetworkParamsWorkspace& params_ws);

	void free_training_data();
	
	void set_params(NetworkParamsWorkspace& params_ws);

	float train(
		const cudaStream_t& stream,
		NetworkParamsWorkspace& params_ws,
		const uint32_t& step,
		const uint32_t& batch_size,
		const uint32_t& n_rays,
		const uint32_t& n_samples,
		const int& aabb_scale,
		const float* random_rgb,
		uint32_t* ray_steps,
		uint32_t* ray_offset,
		float* pos_batch,
		float* dir_batch,
		float* dt_batch,
		float* target_rgba,
		tcnn::network_precision_t* concat_buffer,
		tcnn::network_precision_t* output_buffer
	);

	void inference(
		const cudaStream_t& stream,
		NetworkParamsWorkspace& params_ws,
		const uint32_t& batch_size,
		const int& aabb_scale,
		float* pos_batch,
		float* dir_batch,
		tcnn::network_precision_t* concat_buffer,
		tcnn::network_precision_t* output_buffer,
		const bool& use_color_network = true // if this flag is false, we only run inference on the density network
	);

	size_t get_concat_buffer_width() const {
		return color_network->input_width();
	};

	size_t get_padded_output_width() const {
		return color_network->padded_output_width();
	};

	bool can_train() const {
		return _can_train;
	};


private:

	int aabb_scale = 0;
	uint32_t batch_size = 0;
	bool _can_train = false;

	void update_aabb_scale_if_needed(const int& aabb_scale);

	// Helper context
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrix<float, tcnn::MatrixLayout::RowMajor> density_network_input_matrix;
		tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::MatrixLayout::RowMajor> density_network_output_matrix;

		tcnn::GPUMatrix<float, tcnn::MatrixLayout::RowMajor> direction_encoding_input_matrix;
		tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::MatrixLayout::RowMajor> direction_encoding_output_matrix;
		
		tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::MatrixLayout::RowMajor> color_network_input_matrix;
		tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::MatrixLayout::RowMajor> color_network_output_matrix;
		
		tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::MatrixLayout::RowMajor> density_dL_doutput;
		tcnn::GPUMatrix<tcnn::network_precision_t, tcnn::MatrixLayout::RowMajor> color_dL_doutput;
		tcnn::GPUMatrix<float, tcnn::MatrixLayout::RowMajor> L;

		std::unique_ptr<tcnn::Context> density_ctx;
		std::unique_ptr<tcnn::Context> color_ctx;
	};

	std::unique_ptr<ForwardContext> forward(
		const cudaStream_t& stream,
		const uint32_t& batch_size,
		float* pos_batch,
		float* dir_batch,
		tcnn::network_precision_t* concat_buffer,
		tcnn::network_precision_t* output_buffer
	);

	float calculate_loss(
		const cudaStream_t& stream,
		const uint32_t& batch_size,
		const uint32_t& n_rays
	);

	void optimizer_step(
		const cudaStream_t& stream,
		NetworkParamsWorkspace& params_ws
	);

	void backward(
		const cudaStream_t& stream,
		const std::unique_ptr<NerfNetwork::ForwardContext>& fwd_ctx,
		const uint32_t& batch_size,
		const uint32_t& n_rays,
		const uint32_t& n_samples
	);

	void fused_custom_forward_backward(
		const cudaStream_t& stream,
		const uint32_t& batch_size,
		const uint32_t& n_rays,
		const uint32_t& n_samples,

		const uint32_t* ray_steps,
		const uint32_t* ray_offset,
		const float* random_rgb,
		float* target_rgba,
		float* pos_batch,
		float* dir_batch,
		float* dt_batch,

		tcnn::network_precision_t* concat_buffer,
		tcnn::network_precision_t* output_buffer
	);
	
	void enlarge_workspace_if_needed(const cudaStream_t& stream, const uint32_t& batch_size);
};

TURBO_NAMESPACE_END
