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

	sigma[i] = (T)__expf(fminf((float)density[i] - 1.0f, 11.0f));
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
	
	dL_ddensity[i] = dL_dsigma[i] * __expf(fminf((float)density[i] - 1.0f, 11.0f));
}

// sigma to ray color
__global__ void sigma_to_ray_rgba_forward_kernel(
    uint32_t n_rays,
    uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
    const float* __restrict__ sigma,
    const float* __restrict__ dt,
    const tcnn::network_precision_t* __restrict__ sample_color,
	float* __restrict__ trans,
	float* __restrict__ alpha,
    float* __restrict__ ray_rgba
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;
    
	// offsets
	const uint32_t n_samples = n_samples_per_ray[idx];
	const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

    // local references to sample data
    const float* __restrict__ s_sigma = sigma + sample_offset;
    const float* __restrict__ s_dt = dt + sample_offset;
    
    const tcnn::network_precision_t* __restrict__ s_r = sample_color + sample_offset;
    const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
    const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;

	float* __restrict__ s_trans = trans + sample_offset;
	float* __restrict__ s_alpha = alpha + sample_offset;

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;

    for (int i = 0; i < n_samples; ++i) {
        const float trans = sigma_to_trans(s_sigma, s_dt, i);
        const float alpha = sigma_to_alpha(s_sigma, s_dt, i);
		
		if (trans < 1e-4f) {
			s_trans[i] = 0.0f;
			s_alpha[i] = 0.0f;
			continue;
		}

		s_trans[i] = trans;
		s_alpha[i] = alpha;

        const float weight = trans * alpha;

        r += weight * (float)s_r[i];
        g += weight * (float)s_g[i];
        b += weight * (float)s_b[i];
        a += weight;
    }

    ray_rgba[idx + 0 * batch_size] = r;
    ray_rgba[idx + 1 * batch_size] = g;
    ray_rgba[idx + 2 * batch_size] = b;
    ray_rgba[idx + 3 * batch_size] = a;
}

// sigma to ray color backward
// gets dL/dsigma = dL/dray_color * dray_color/dsigma

__global__ void sigma_to_ray_rgba_backward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const uint32_t* __restrict__ n_samples_per_ray,
    const uint32_t* __restrict__ n_samples_cum,
    const float* __restrict__ sigma,
    const float* __restrict__ dt,
	const float* __restrict__ trans,
	const float* __restrict__ alpha,
    const tcnn::network_precision_t* __restrict__ sample_color,
    const float* __restrict__ dL_dR,
    float* __restrict__ dL_dsigma,
	float* __restrict__ dL_dcolor
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

    // offsets
    const uint32_t n_samples = n_samples_per_ray[idx];
    const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

    // local references to sample data
    const float* __restrict__ s_sigma = sigma + sample_offset;
    const float* __restrict__ s_dt = dt + sample_offset;

    const tcnn::network_precision_t* __restrict__ s_r = sample_color + sample_offset;
    const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
    const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;
	
	const float* __restrict__ s_trans = trans + sample_offset;
	const float* __restrict__ s_alpha = alpha + sample_offset;

    const float dL_dR_r = dL_dR[idx + 0 * batch_size];
	const float dL_dR_g = dL_dR[idx + 1 * batch_size];
	const float dL_dR_b = dL_dR[idx + 2 * batch_size];
	const float dL_dR_a = dL_dR[idx + 3 * batch_size];

    float* __restrict__ s_dL_dsigma = dL_dsigma + sample_offset;
	float* __restrict__ s_dL_dcolor = dL_dcolor + sample_offset;

   for (int i = 0; i < n_samples; ++i) {
		if (s_trans[i] < 1e-4f) {
			s_dL_dsigma[i] = 0.0f;
			s_dL_dcolor[i + 0 * batch_size] = 0.0f;
			s_dL_dcolor[i + 1 * batch_size] = 0.0f;
			s_dL_dcolor[i + 2 * batch_size] = 0.0f;
			continue;
		}

        float cumsum_r = 0.0f;
        float cumsum_g = 0.0f;
        float cumsum_b = 0.0f;
        float cumsum_a = 0.0f;

        for (int j = i + 1; j < n_samples; ++j) {
            const float k = -s_dt[i] * s_trans[j] * s_alpha[j];

            cumsum_r += k * (float)s_r[j];
            cumsum_g += k * (float)s_g[j];
            cumsum_b += k * (float)s_b[j];
			cumsum_a += k;
        }
        
        const float c = s_trans[i] * s_dt[i] * (1.0f - s_alpha[i]);

        const float dRr_dsigma = cumsum_r + c * (float)s_r[i];
        const float dRg_dsigma = cumsum_g + c * (float)s_g[i];
        const float dRb_dsigma = cumsum_b + c * (float)s_b[i];
		const float dRa_dsigma = cumsum_a + c;

        s_dL_dsigma[i] = dL_dR_r * dRr_dsigma + dL_dR_g * dRg_dsigma + dL_dR_b * dRb_dsigma + dL_dR_a * dRa_dsigma;
		
		const float s_weight_i = s_trans[i] * s_alpha[i];

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
