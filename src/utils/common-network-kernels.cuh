#pragma once

#include "../common.h"

#include <stdint.h>

template <typename T>
inline __device__ float sigma_to_alpha(
    const T* __restrict__ sigma,
    const float* __restrict__ dt,
    const int& i
) {
    return 1.0f - __expf(-(float)sigma[i] * dt[i]);
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

	alpha[i] = sigma_to_alpha(sigma, dt, i);
}


