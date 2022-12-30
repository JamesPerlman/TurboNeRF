#pragma once

#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct NerfNetwork {
	std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> direction_encoding;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> density_network;
	std::shared_ptr<tcnn::Network<tcnn::network_precision_t>> color_network;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;
	
	NerfNetwork();
	
	void train(cudaStream_t stream, uint32_t batch_size, float* pos_batch, float* dir_batch, float* rgb_batch);

private:

	// full-precision params buffers
	tcnn::GPUMemory<float> density_network_params_fp;
	tcnn::GPUMemory<tcnn::network_precision_t> density_network_params_hp;
	tcnn::GPUMemory<tcnn::network_precision_t> density_network_gradients_hp;

	tcnn::GPUMemory<float> color_network_params_fp;
	tcnn::GPUMemory<tcnn::network_precision_t> color_network_params_hp;
	tcnn::GPUMemory<tcnn::network_precision_t> color_network_gradients_hp;

	tcnn::GPUMemory<tcnn::network_precision_t> color_network_input;
	tcnn::GPUMemory<tcnn::network_precision_t> color_network_output;

	uint32_t previous_batch_size = 0;
	
	void initialize_params_and_gradients();
	void enlarge_batch_memory_if_needed(uint32_t batch_size);

	void forward(cudaStream_t stream, uint32_t batch_size, float* pos_batch, float* dir_batch);
};

NRC_NAMESPACE_END
