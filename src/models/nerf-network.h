#pragma once

#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct NerfNetwork {
	std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> direction_encoding;
	std::shared_ptr<tcnn::cpp::Module> density_network;
	std::shared_ptr<tcnn::cpp::Module> color_network;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;
	
	NerfNetwork();
	
	void train(cudaStream_t stream, uint32_t batch_size, float* rgb_batch, float* dir_batch);

private:

	// full-precision params buffers
	tcnn::GPUMemory<float> density_network_params_fp;
	tcnn::GPUMemory<tcnn::network_precision_t> color_network_input;

	uint32_t previous_batch_size = 0;
	
	void enlarge_batch_memory_if_needed(uint32_t batch_size);
};

NRC_NAMESPACE_END
