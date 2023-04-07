#pragma once

#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"
#include "../utils/nerf-constants.cuh"
#include "common-network-kernels.cuh"

TURBO_NAMESPACE_BEGIN

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

template <typename T>
inline __device__ float density_to_sigma(const T& density) {
	return __expf((float)density - 1.0f);
}

/**
 * Apply exponential scaling to density network output
 * (log-space density!)
 */
// exp_sigma(sigma_network_output) = exp(sigma_network_output - 1.0f)

template <typename T>
__global__ void density_to_sigma_forward_kernel(
	const uint32_t n_samples,
	const tcnn::network_precision_t* __restrict__ density,
	T* __restrict__ sigma
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_samples) return;

	sigma[i] = (T)density_to_sigma(fminf(density[i], 12.0f));
}

// this computes dL/ddensity = dL/dsigma * dsigma/ddensity and applies it back to the original density
__global__ void density_to_sigma_backward_kernel(
	uint32_t n_samples,
	const float* __restrict__ sigma,
	const float* __restrict__ dL_dsigma,
	float* __restrict__ dL_ddensity
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_samples) return;
	
	dL_ddensity[i] = dL_dsigma[i] * sigma[i];
}

// sigma to ray color
__global__ void sigma_to_ray_rgba_forward_kernel(
    uint32_t n_rays,
    uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_buf,
	const uint32_t* __restrict__ ray_offset_buf,
    const tcnn::network_precision_t* __restrict__ sample_rgb_buf,
	const float* __restrict__ alpha_buf,
    float* __restrict__ ray_rgba_buf
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;
    
	// offsets
	const uint32_t n_samples = n_samples_buf[idx];
	const uint32_t sample_offset = ray_offset_buf[idx];

    // local references to sample data
    const tcnn::network_precision_t* __restrict__ s_r = sample_rgb_buf + sample_offset;
    const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
    const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;

	const float* __restrict__ s_alpha = alpha_buf + sample_offset;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;

	float trans = 1.0f;

    for (int i = 0; i < n_samples; ++i) {
        const float alpha = s_alpha[i];
		
        const float weight = trans * alpha;

        r += weight * (float)s_r[i];
        g += weight * (float)s_g[i];
        b += weight * (float)s_b[i];
        a += weight;

		trans *= (1.0f - alpha);

		if (trans < NeRFConstants::min_transmittance) {
			break;
		}
    }

	const uint32_t idx_offset_0 = idx;
	const uint32_t idx_offset_1 = idx_offset_0 + batch_size;
	const uint32_t idx_offset_2 = idx_offset_1 + batch_size;
	const uint32_t idx_offset_3 = idx_offset_2 + batch_size;

    ray_rgba_buf[idx_offset_0] = r;
    ray_rgba_buf[idx_offset_1] = g;
    ray_rgba_buf[idx_offset_2] = b;
    ray_rgba_buf[idx_offset_3] = a;
}

// sigma to ray color backward
// gets dL/dsigma = dL/dray_color * dray_color/dsigma

__global__ void sigma_to_ray_rgba_backward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const uint32_t* __restrict__ n_samples_buf,
    const uint32_t* __restrict__ ray_offset_buf,
    const float* __restrict__ dt_buf,
	const float* __restrict__ alpha_buf,
    const tcnn::network_precision_t* __restrict__ sample_rgb_buf,
	const float* __restrict__ random_rgb,
	const float* __restrict__ ray_rgba_buf,
    const float* __restrict__ dL_dR_buf,
    float* __restrict__ dL_dsigma_buf,
	float* __restrict__ dL_dcolor_buf
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

    // offsets
    const uint32_t n_samples = n_samples_buf[idx];
    const uint32_t sample_offset = ray_offset_buf[idx];

	const uint32_t idx_offset_0 = idx;
	const uint32_t idx_offset_1 = idx_offset_0 + batch_size;
	const uint32_t idx_offset_2 = idx_offset_1 + batch_size;
	const uint32_t idx_offset_3 = idx_offset_2 + batch_size;

    // local references to sample data
    const float* __restrict__ s_dt = dt_buf + sample_offset;

    const tcnn::network_precision_t* __restrict__ s_r = sample_rgb_buf + sample_offset;
    const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
    const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;
	
	const float* __restrict__ s_alpha = alpha_buf + sample_offset;

    const float dL_dR_r = dL_dR_buf[idx_offset_0];
	const float dL_dR_g = dL_dR_buf[idx_offset_1];
	const float dL_dR_b = dL_dR_buf[idx_offset_2];

    float* __restrict__ s_dL_dsigma = dL_dsigma_buf + sample_offset;

	float* __restrict__ s_dL_dcolor_r = dL_dcolor_buf + sample_offset;
	float* __restrict__ s_dL_dcolor_g = s_dL_dcolor_r + batch_size;
	float* __restrict__ s_dL_dcolor_b = s_dL_dcolor_g + batch_size;

	// we need to invert the random bg compositing operation to obtain the original color
	// we could potentially save the original rgba in a buffer to avoid this operation

	const float rand_r = random_rgb[idx_offset_0];
	const float rand_g = random_rgb[idx_offset_1];
	const float rand_b = random_rgb[idx_offset_2];

	const float ray_r = ray_rgba_buf[idx_offset_0];
	const float ray_g = ray_rgba_buf[idx_offset_1];
	const float ray_b = ray_rgba_buf[idx_offset_2];
	const float ray_a = ray_rgba_buf[idx_offset_3];

	const float dr = ray_r - rand_r;
	const float dg = ray_g - rand_g;
	const float db = ray_b - rand_b;

	float cumsum_r = ray_r;
	float cumsum_g = ray_g;
	float cumsum_b = ray_b;
	float cumsum_a = ray_a;

	float trans = 1.0f;
	for (int i = 0; i < n_samples; ++i) {
		const float r = s_r[i];
		const float g = s_g[i];
		const float b = s_b[i];

		const float dt = s_dt[i];
		const float alpha = s_alpha[i];
		
		const float weight = trans * alpha;
        const float k = (trans - weight);

		cumsum_r -= weight * r;
		cumsum_g -= weight * g;
		cumsum_b -= weight * b;
		cumsum_a -= weight;

        float dRr_dsigma = dt * (k * r - cumsum_r);
        float dRg_dsigma = dt * (k * g - cumsum_g);
        float dRb_dsigma = dt * (k * b - cumsum_b);
		float dRa_dsigma = dt * (k * 1 - cumsum_a);

		dRr_dsigma = dRr_dsigma * ray_a + dr * dRa_dsigma;
		dRg_dsigma = dRg_dsigma * ray_a + dg * dRa_dsigma;
		dRb_dsigma = dRb_dsigma * ray_a + db * dRa_dsigma;

        s_dL_dsigma[i] = dL_dR_r * dRr_dsigma + dL_dR_g * dRg_dsigma + dL_dR_b * dRb_dsigma;

		const float dR_dcolor = weight * ray_a;
		
		s_dL_dcolor_r[i] = dL_dR_r * dR_dcolor;
		s_dL_dcolor_g[i] = dL_dR_g * dR_dcolor;
		s_dL_dcolor_b[i] = dL_dR_b * dR_dcolor;
		
		trans *= 1.0f - alpha;

		if (trans < NeRFConstants::min_transmittance) {
			break;
		}
    }
}

// smooth L1 loss helpers (beta = 1.0f)
inline __device__ float smooth_l1_loss_forward(const float& x) {
	const float absx = fabsf(x);
	return absx < 1.0f
		? 0.5f * x * x
		: absx - 0.5f;
}

// TODO: put these in a separate file
inline __device__ float smooth_l1_loss_backward(const float& x) {
	return fabsf(x) < 1.0f
		? x
		: copysignf(1.0f, x);
}

// RGBA to loss
__global__ void ray_rgba_to_loss_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const float* __restrict__ random_rgb,
	const float* __restrict__ ray_rgba,
	const float* __restrict__ target_rgba,
	float* __restrict__ sse_loss
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= batch_size) return;

	const uint32_t r_idx = idx;
	const uint32_t g_idx = r_idx + batch_size;
	const uint32_t b_idx = g_idx + batch_size;
	const uint32_t a_idx = b_idx + batch_size;

	if (idx >= n_rays) {
		sse_loss[r_idx] = 0.0f;
		sse_loss[g_idx] = 0.0f;
		sse_loss[b_idx] = 0.0f;
		sse_loss[a_idx] = 0.0f;
		return;
	}

	// https://github.com/cheind/pure-torch-ngp/blob/develop/torchngp/training.py#L301-L314
	// mixing random colors with predicted and ground truth colors encourages the network to learn empty space
	// THANK YOU CHEIND

	// TODO: Clean this up.  Here is a suboptimal place to change this buffer data.
	// Ideally ground truth colors should be mixed during batch generation.
	// Predicted colors should be mixed during ray accumulation.
	// "Sir, this is a loss function."

	const float rand_r = random_rgb[r_idx];
	const float rand_g = random_rgb[g_idx];
	const float rand_b = random_rgb[b_idx];

	const float gt_a = target_rgba[a_idx];
	const float gt_a_comp = 1.0f - gt_a;
	const float gt_r = target_rgba[r_idx] * gt_a + rand_r * gt_a_comp;
	const float gt_g = target_rgba[g_idx] * gt_a + rand_g * gt_a_comp;
	const float gt_b = target_rgba[b_idx] * gt_a + rand_b * gt_a_comp;

	const float ray_a = ray_rgba[a_idx];
	const float ray_a_comp = 1.0f - ray_a;
	const float ray_r = ray_rgba[r_idx] * ray_a + rand_r * ray_a_comp;
	const float ray_g = ray_rgba[g_idx] * ray_a + rand_g * ray_a_comp;
	const float ray_b = ray_rgba[b_idx] * ray_a + rand_b * ray_a_comp;

	sse_loss[r_idx] = smooth_l1_loss_forward(ray_r - gt_r);
	sse_loss[g_idx] = smooth_l1_loss_forward(ray_g - gt_g);
	sse_loss[b_idx] = smooth_l1_loss_forward(ray_b - gt_b);
	sse_loss[a_idx] = smooth_l1_loss_forward(ray_a - gt_a);
}

// dL/dR = (1/4) * (dL/dR_r + dL/dR_g + dL/dR_b + dL/dR_a)
__global__ void ray_rgba_to_loss_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const float inv_3nrays,
	const float* __restrict__ random_rgb,
	const float* __restrict__ target_rgba,
	const float* __restrict__ ray_rgba,
	float* __restrict__ dL_dR
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) return;

	const uint32_t r_idx = idx;
	const uint32_t g_idx = r_idx + batch_size;
	const uint32_t b_idx = g_idx + batch_size;
	const uint32_t a_idx = b_idx + batch_size;

	const float rand_r = random_rgb[r_idx];
	const float rand_g = random_rgb[g_idx];
	const float rand_b = random_rgb[b_idx];

	const float gt_a = target_rgba[a_idx];
	const float gt_a_comp = 1.0f - gt_a;
	const float gt_r = target_rgba[r_idx] * gt_a + rand_r * gt_a_comp;
	const float gt_g = target_rgba[g_idx] * gt_a + rand_g * gt_a_comp;
	const float gt_b = target_rgba[b_idx] * gt_a + rand_b * gt_a_comp;

	const float ray_a = ray_rgba[a_idx];
	const float ray_a_comp = 1.0f - ray_a;
	const float ray_r = ray_rgba[r_idx] * ray_a + rand_r * ray_a_comp;
	const float ray_g = ray_rgba[g_idx] * ray_a + rand_g * ray_a_comp;
	const float ray_b = ray_rgba[b_idx] * ray_a + rand_b * ray_a_comp;

	dL_dR[r_idx] = inv_3nrays * smooth_l1_loss_backward(ray_r - gt_r);
	dL_dR[g_idx] = inv_3nrays * smooth_l1_loss_backward(ray_g - gt_g);
	dL_dR[b_idx] = inv_3nrays * smooth_l1_loss_backward(ray_b - gt_b);
}

TURBO_NAMESPACE_END
