#pragma once

#define NRC_NAMESPACE_BEGIN namespace nrc {
#define NRC_NAMESPACE_END }

#define NRC_HOST_DEVICE __host__ __device__

#define CUDA_CHECK_THROW(call) { \
	cudaError_t err = call; \
	if (err != cudaSuccess) { \
		throw std::runtime_error(cudaGetErrorString(err)); \
	} \
}
