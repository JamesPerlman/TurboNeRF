#pragma once

#include "../common.h"

#include <stdint.h>

template <typename T>
inline __device__ float density_to_sigma(const T& density) {
	return __expf((float)density - 1.0f);
}

inline __device__ float sigma_to_alpha(
	const float& sigma,
	const float& dt
) {
	return 1.0f - __expf(-(float)sigma * dt);
}

template <typename T>
inline __device__ float density_to_alpha(
	const T& density,
	const float& dt
) {
	return sigma_to_alpha(density_to_sigma(density), dt);
}

template <typename T>
__global__ void sigma_to_alpha_forward_kernel(
	uint32_t n_samples,
	const T* __restrict__ sigma,
	const float* __restrict__ dt,
	float* __restrict__ alpha
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_samples) return;

	alpha[i] = sigma_to_alpha(sigma[i], dt[i]);
}

template <int N_CHANNELS, bool ACCUMULATE>
__global__ void copy_gradients_kernel(
	const uint32_t n_elements,
	const uint32_t data_stride,
	const float scale,
	const float* __restrict__ input,
	tcnn::network_precision_t* __restrict__ output
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_elements) return;

	#pragma unroll
	for (int j = 0; j < N_CHANNELS; ++j) {
		if (ACCUMULATE)
			output[idx] = (tcnn::network_precision_t)(scale * input[idx]) + output[idx];
		else
			output[idx] = (tcnn::network_precision_t)(scale * input[idx]);
		
		idx += data_stride;
	}
}

