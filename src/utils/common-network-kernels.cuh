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


