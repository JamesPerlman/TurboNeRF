#pragma once

#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"

NRC_NAMESPACE_BEGIN

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

/**
 * All of this code is deliberately under-optimized for readability.  Will fuse kernels and optimize later.
 */

template <typename T>
inline __device__ float density_to_sigma(const T& density) {
	return __expf((float)density);
}

inline __device__ float sigma_to_trans(
    const float* __restrict__ sigma,
    const float* __restrict__ dt,
    const int& i
) {
    float sigma_dt = 0.0f;
    for (int j = 0; j < i; ++j) {
        sigma_dt += sigma[j] * dt[j];
    }
    return __expf(-sigma_dt);
}

inline __device__ float sigma_to_alpha(
    const float* __restrict__ sigma,
    const float* __restrict__ dt,
    const int& i
) {
    return 1.0f - __expf(-sigma[i] * dt[i]);
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

	sigma[i] = (T)density_to_sigma(density[i]);
}

// this computes dL/ddensity = dL/dsigma * dsigma/ddensity and applies it back to the original density
__global__ void density_to_sigma_backward_kernel(
	uint32_t n_samples,
	const tcnn::network_precision_t* __restrict__ density,
	const float* __restrict__ dL_dsigma,
	float* __restrict__ dL_ddensity
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_samples) return;
	
	dL_ddensity[i] = dL_dsigma[i] * density_to_sigma(density[i]);
}

// sigma to ray color
__global__ void sigma_to_ray_rgba_forward_kernel(
    uint32_t n_rays,
    uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_buf,
	const uint32_t* __restrict__ n_samples_cum_buf,
    const float* __restrict__ sigma_buf,
    const float* __restrict__ dt_buf,
    const tcnn::network_precision_t* __restrict__ sample_rgb_buf,
	float* __restrict__ trans_buf,
	float* __restrict__ alpha_buf,
    float* __restrict__ ray_rgba_buf
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;
    
	// offsets
	const uint32_t n_samples = n_samples_buf[idx];
	const uint32_t sample_offset = n_samples_cum_buf[idx] - n_samples;

    // local references to sample data
    const float* __restrict__ s_sigma = sigma_buf + sample_offset;
    const float* __restrict__ s_dt = dt_buf + sample_offset;
    
    const tcnn::network_precision_t* __restrict__ s_r = sample_rgb_buf + sample_offset;
    const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
    const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;

	float* __restrict__ s_trans = trans_buf + sample_offset;
	float* __restrict__ s_alpha = alpha_buf + sample_offset;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;

	float trans = 1.0f;

    for (int i = 0; i < n_samples; ++i) {
        const float alpha = sigma_to_alpha(s_sigma, s_dt, i);

		s_trans[i] = trans;
		s_alpha[i] = alpha;

		if (trans < 1e-4f) {
			break;
		}
		
        const float weight = trans * alpha;

        r += weight * (float)s_r[i];
        g += weight * (float)s_g[i];
        b += weight * (float)s_b[i];
        a += weight;

		trans *= (1.0f - alpha);

    }

    ray_rgba_buf[idx + 0 * batch_size] = r;
    ray_rgba_buf[idx + 1 * batch_size] = g;
    ray_rgba_buf[idx + 2 * batch_size] = b;
    ray_rgba_buf[idx + 3 * batch_size] = a;
}

// sigma to ray color backward
// gets dL/dsigma = dL/dray_color * dray_color/dsigma

__global__ void sigma_to_ray_rgba_backward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const uint32_t* __restrict__ n_samples_buf,
    const uint32_t* __restrict__ n_samples_cum_buf,
    const float* __restrict__ sigma_buf,
    const float* __restrict__ dt_buf,
	const float* __restrict__ trans_buf,
	const float* __restrict__ alpha_buf,
    const tcnn::network_precision_t* __restrict__ sample_rgb_buf,
	const float* __restrict__ ray_rgba_buf,
    const float* __restrict__ dL_dR_buf,
    float* __restrict__ dL_dsigma_buf,
	float* __restrict__ dL_dcolor_buf
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

    // offsets
    const uint32_t n_samples = n_samples_buf[idx];
    const uint32_t sample_offset = n_samples_cum_buf[idx] - n_samples;

	const uint32_t idx_offset_0 = idx;
	const uint32_t idx_offset_1 = idx_offset_0 + batch_size;
	const uint32_t idx_offset_2 = idx_offset_1 + batch_size;
	const uint32_t idx_offset_3 = idx_offset_2 + batch_size;

    // local references to sample data
    const float* __restrict__ s_sigma = sigma_buf + sample_offset;
    const float* __restrict__ s_dt = dt_buf + sample_offset;

    const tcnn::network_precision_t* __restrict__ s_r = sample_rgb_buf + sample_offset;
    const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
    const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;
	
	const float* __restrict__ s_trans = trans_buf + sample_offset;
	const float* __restrict__ s_alpha = alpha_buf + sample_offset;

    const float dL_dR_r = dL_dR_buf[idx_offset_0];
	const float dL_dR_g = dL_dR_buf[idx_offset_1];
	const float dL_dR_b = dL_dR_buf[idx_offset_2];
	const float dL_dR_a = dL_dR_buf[idx_offset_3];

    float* __restrict__ s_dL_dsigma = dL_dsigma_buf + sample_offset;
	float* __restrict__ s_dL_dcolor = dL_dcolor_buf + sample_offset;

	float cumsum_r = ray_rgba_buf[idx_offset_0];
	float cumsum_g = ray_rgba_buf[idx_offset_1];
	float cumsum_b = ray_rgba_buf[idx_offset_2];
	float cumsum_a = ray_rgba_buf[idx_offset_3];

	for (int i = 0; i < n_samples; ++i) {
		const float s_trans_i = s_trans[i];

		if (s_trans_i < 1e-4f) {
			break;
		}

		const float s_alpha_i = s_alpha[i];
		const float s_dt_i = s_dt[i];
		const float s_weight_i = s_trans_i * s_alpha_i;
        const float c = s_trans_i * s_dt_i * (1.0f - s_alpha_i);

		
		cumsum_r -= s_weight_i * (float)s_r[i];
		cumsum_g -= s_weight_i * (float)s_g[i];
		cumsum_b -= s_weight_i * (float)s_b[i];
		cumsum_a -= s_weight_i;


        const float dRr_dsigma = -s_dt_i * cumsum_r + c * (float)s_r[i];
        const float dRg_dsigma = -s_dt_i * cumsum_g + c * (float)s_g[i];
        const float dRb_dsigma = -s_dt_i * cumsum_b + c * (float)s_b[i];
		const float dRa_dsigma = -s_dt_i * cumsum_a + c;

        s_dL_dsigma[i] = dL_dR_r * dRr_dsigma + dL_dR_g * dRg_dsigma + dL_dR_b * dRb_dsigma + dL_dR_a * dRa_dsigma;


		s_dL_dcolor[i + 0 * batch_size] = dL_dR_r * s_weight_i;
		s_dL_dcolor[i + 1 * batch_size] = dL_dR_g * s_weight_i;
		s_dL_dcolor[i + 2 * batch_size] = dL_dR_b * s_weight_i;

    }
}

// RGBA to loss
__global__ void ray_rgba_to_loss_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const float* __restrict__ ray_rgba,
	const float* __restrict__ target_rgba,
	float* __restrict__ sse_loss
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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

	const float dr = ray_rgba[r_idx] - target_rgba[r_idx];
	const float dg = ray_rgba[g_idx] - target_rgba[g_idx];
	const float db = ray_rgba[b_idx] - target_rgba[b_idx];
	const float da = ray_rgba[a_idx] - target_rgba[a_idx];

	sse_loss[r_idx] = (dr * dr);
	sse_loss[g_idx] = (dg * dg);
	sse_loss[b_idx] = (db * db);
	sse_loss[a_idx] = (da * da);
}

// dL/dR = (1/4) * (dL/dR_r + dL/dR_g + dL/dR_b + dL/dR_a)
__global__ void ray_rgba_to_loss_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const float inv_2nrays,
	const float* __restrict__ ray_rgba,
	const float* __restrict__ target_rgba,
	float* __restrict__ dL_dR
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) {
		return;
	}

	const uint32_t r_idx = idx;
	const uint32_t g_idx = r_idx + batch_size;
	const uint32_t b_idx = g_idx + batch_size;
	const uint32_t a_idx = b_idx + batch_size;

	const float dr = ray_rgba[r_idx] - target_rgba[r_idx];
	const float dg = ray_rgba[g_idx] - target_rgba[g_idx];
	const float db = ray_rgba[b_idx] - target_rgba[b_idx];
	const float da = ray_rgba[a_idx] - target_rgba[a_idx];

	dL_dR[r_idx] = inv_2nrays * dr;
	dL_dR[g_idx] = inv_2nrays * dg;
	dL_dR[b_idx] = inv_2nrays * db;
	dL_dR[a_idx] = inv_2nrays * da;
}

NRC_NAMESPACE_END
