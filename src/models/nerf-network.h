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
	
	void enlarge_batch_memory_if_needed(uint32_t batch_size);
	void forward(cudaStream_t stream, uint32_t batch_size, float* pos_batch, float* dir_batch);

	/**
	 * The density MLP maps the hash encoded position y = enc(x; 𝜃)
	 * to 16 output values, the first of which we treat as log-space density
	 * https://arxiv.org/abs/2201.05989 - page 9
	 */
	const tcnn::network_precision_t* get_log_space_density() const {
		// The output of the density network is just a pointer to the color network's input buffer.
		return color_network_input.data();
	}

	const tcnn::network_precision_t* get_color_network_output() const {
		return color_network_output.data();
	}

	const size_t get_color_network_padded_output_width() const {
		return color_network->padded_output_width();
	}

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
	
	void initialize_params_and_gradients();
};

NRC_NAMESPACE_END
