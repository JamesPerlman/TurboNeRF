#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <tiny-cuda-nn/common.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN


/**
 * Accumulate the sample colors by ray.
 */

__global__ void accumulate_ray_colors_from_samples_kernel(
	uint32_t n_rays, // n_rays is not necessarily the batch size here
	uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,

	// these input buffers organized based on the cumulative steps
	const tcnn::network_precision_t* __restrict__ network_rgb,
	const tcnn::network_precision_t* __restrict__ network_sigma,
	const float* __restrict__ sample_dt,

	// output buffer organized by ray index
	float* __restrict__ ray_rgba
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_rays) {
		return;
	}

	// Grab local references to global data
	const uint32_t n_samples = n_samples_per_ray[i];
	const uint32_t sample_offset = n_samples_cum[i] - n_samples;

	const tcnn::network_precision_t* __restrict__ r = network_rgb + sample_offset;
	const tcnn::network_precision_t* __restrict__ g = r + batch_size;
	const tcnn::network_precision_t* __restrict__ b = g + batch_size;

	const tcnn::network_precision_t* __restrict__ sigma = network_sigma + sample_offset;

	const float* __restrict__ dt = sample_dt + sample_offset;

	// Values to accumulate samples into
	float ray_r = 0.0f;
	float ray_g = 0.0f;
	float ray_b = 0.0f;
	float ray_a = 0.0f;

	float sigma_cumsum = 0.0f;

	// Accumulate samples
	for (int j = 0; j < n_samples; ++j) {
		// thank you NerfAcc
		// TODO: understand what this does
		const float sigma_j = (float)sigma[j];
		const float alpha = 1.0f - expf(-sigma_j);
		const float transmittance = expf(-sigma_cumsum);
		sigma_cumsum += sigma_j * dt[j];

		const float weight = alpha * transmittance;

		// accumulate the color
		ray_r += weight * (float)r[j];
		ray_g += weight * (float)g[j];
		ray_b += weight * (float)b[j];
		ray_a += weight;
	}
	
	// write out the accumulated ray color
	ray_rgba[i + 0 * batch_size] = ray_r;
	ray_rgba[i + 1 * batch_size] = ray_g;
	ray_rgba[i + 2 * batch_size] = ray_b;
	ray_rgba[i + 3 * batch_size] = ray_a;
}

// Calculate the loss
__global__ void calculate_mse_loss_per_ray_kernel(
	uint32_t n_pixels,
	uint32_t batch_size, // aka data stride
	const float* __restrict__ network_rgba, // this is the output of the network, accumulated by ray
	const float* __restrict__ target_rgba, // the ground-truth pixel colors for each ray
    // const float loss_scale,
	float* __restrict__ out_loss
    // tcnn::network_precision_t* __restrict__ out_grad
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_pixels) {
		return;
	}

	// Grab local references to global data
	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;
	const uint32_t i_offset_3 = i_offset_2 + batch_size;

	// Calculate the loss
	const float dr = (float)network_rgba[i_offset_0] - target_rgba[i_offset_0];
	const float dg = (float)network_rgba[i_offset_1] - target_rgba[i_offset_1];
	const float db = (float)network_rgba[i_offset_2] - target_rgba[i_offset_2];
	const float da = (float)network_rgba[i_offset_3] - target_rgba[i_offset_3];

	// mean squared error per ray - still needs to be divided by n_rays
	out_loss[i_offset_0] = 0.25f * (dr *dr + dg * dg + db * db + da * da);

    // out_grad[i_offset_0] = (tcnn::network_precision_t)(loss_scale * 2.0f * dr / n_pixels);
    // out_grad[i_offset_1] = (tcnn::network_precision_t)(loss_scale * 2.0f * dg / n_pixels);
    // out_grad[i_offset_2] = (tcnn::network_precision_t)(loss_scale * 2.0f * db / n_pixels);
	// out_grad[i_offset_3] = (tcnn::network_precision_t)(loss_scale * 2.0f * da / n_pixels);
}


NRC_NAMESPACE_END