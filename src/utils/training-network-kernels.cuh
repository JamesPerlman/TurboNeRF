#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

/**
 * THESE ARE DELIBERATELY UNDER-OPTIMIZED BECAUSE I AM LEARNING HOW TO BACKPROPAGATE
 */

/**
 * Transmittance from activated sigma
 */

// calculates transmittance from sigma
__global__ void sigma_to_alpha_and_transmittance_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	float* __restrict__ alpha,
	float* __restrict__ transmittance
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= n_rays) return;

	// offsets
	const uint32_t n_samples = n_samples_per_ray[idx];
	const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

	// local references to sample data
	const float* __restrict__ s_dt = dt + sample_offset;
	const float* __restrict__ s_sigma = sigma + sample_offset;
	float* __restrict__ s_alpha = alpha + sample_offset;
	float* __restrict__ s_trans = transmittance + sample_offset;

	for (int i = 0; i < n_samples; ++i) {
		float sigma_cumsum = 0.0f;
		for (int j = 0; j < i; ++j) {
			sigma_cumsum += s_sigma[j] * s_dt[j];
		}
		s_alpha[i] = 1.0f - __expf(-s_sigma[i] * s_dt[i]);
		s_trans[i] = __expf(-sigma_cumsum);
	}
}


// calculates dL/dsigma = dL/dtransmittance * dtransmittance/dsigma + dL/dalpha * dalpha/dsigma
__global__ void sigma_to_alpha_and_transmittance_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	const float* __restrict__ alpha,
	const float* __restrict__ transmittance,
	const float* __restrict__ dL_dalpha,
	const float* __restrict__ dL_dtransmittance,
	float* __restrict__ dL_dsigma
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) return;

	// offsets
	const uint32_t n_samples = n_samples_per_ray[idx];
	const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

	// local references to sample data
	const float* __restrict__ s_dt = dt + sample_offset;
	const float* __restrict__ s_sigma = sigma + sample_offset;
	const float* __restrict__ s_alpha = alpha + sample_offset;
	const float* __restrict__ s_trans = transmittance + sample_offset;
	const float* __restrict__ s_dL_dalpha = dL_dalpha + sample_offset;
	const float* __restrict__ s_dL_dtrans = dL_dtransmittance + sample_offset;
	float* __restrict__ s_dL_dsigma = dL_dsigma + sample_offset;

    for (int i = 0; i < n_samples; ++i) {
        float dtrans_dsigma_sum = 0.0f;
        for (int j = i + 1; j < n_samples; ++j) {
            float cumsum_inner = 0.0;
            for (int k = 0; k < j; ++k) {
                cumsum_inner += s_sigma[k] * s_dt[k];
            }
            dtrans_dsigma_sum += __expf(-cumsum_inner) * s_dL_dtrans[j]; //?
        }
		const float dL_dtrans_dtrans_dsigma = -s_dt[i] * dtrans_dsigma_sum;
		const float dalpha_dsigma = s_dt[i] * (1.0f - s_alpha[i]);
        s_dL_dsigma[i] = dL_dtrans_dtrans_dsigma + dalpha_dsigma * s_dL_dalpha[i];
    }
}

// calculates weight = alpha * transmittance
__global__ void alpha_and_transmittance_to_weight_forward_kernel(
	const uint32_t n_samples,
	const uint32_t batch_size,
	const float* __restrict__ alpha,
	const float* __restrict__ transmittance,
	float* __restrict__ weight
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples) return;

	weight[idx] = alpha[idx] * transmittance[idx];
}

// calculates dL/dalpha = dL/dweight * dweight/dalpha
// calculates dL/dtransmittance = dL/dweight * dweight/dtransmittance
__global__ void alpha_and_transmittance_to_weight_backward_kernel(
	const uint32_t n_samples,
	const uint32_t batch_size,
	const float* __restrict__ alpha,
	const float* __restrict__ transmittance,
	const float* __restrict__ dL_dweight,
	float* __restrict__ dL_dalpha,
	float* __restrict__ dL_dtransmittance
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_samples) return;

	const float dweight_dalpha = transmittance[idx];
	dL_dalpha[idx] = dL_dweight[idx] * dweight_dalpha;

	const float dweight_dtransmittance = alpha[idx];
	dL_dtransmittance[idx] = dL_dweight[idx] * dweight_dtransmittance;
}

/**
 * Weight to color (for a single channel)
 */

// This takes in samples and weights and accumulates their values into a ray's color channel

__global__ void weight_to_ray_rgba_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ weight,
	const tcnn::network_precision_t* __restrict__ sample_rgb,
	float* __restrict__ ray_rgba
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) return;

	// offsets
	const uint32_t n_samples = n_samples_per_ray[idx];
	const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

	const float* __restrict__ s_weight = weight + sample_offset;

	const tcnn::network_precision_t* __restrict__ r_sample = sample_rgb + sample_offset;
	const tcnn::network_precision_t* __restrict__ g_sample = r_sample + batch_size;
	const tcnn::network_precision_t* __restrict__ b_sample = g_sample + batch_size;

	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;
	float a = 0.0f;

	for (int i = 0; i < n_samples; ++i) {
		r += s_weight[i] * (float)r_sample[i];
		g += s_weight[i] * (float)g_sample[i];
		b += s_weight[i] * (float)b_sample[i];
		a += s_weight[i];
	}

	ray_rgba[idx + 0 * batch_size] = r;
	ray_rgba[idx + 1 * batch_size] = g;
	ray_rgba[idx + 2 * batch_size] = b;
	ray_rgba[idx + 3 * batch_size] = a;
}

// calculates dL/weight = dL/dR * dR/weight
// calculates dL/dcolor = dL/dR * dR/dcolor
__global__ void weight_to_ray_rgba_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ weight,
	const tcnn::network_precision_t* __restrict__ network_rgb,
	const float* __restrict__ dL_dR,
	float* __restrict__ dL_dweight,
	float* __restrict__ dL_dcolor
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) return;

	// offsets
	const uint32_t n_samples = n_samples_per_ray[idx];
	const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

	const float* __restrict__ s_weight = weight + sample_offset;

	const tcnn::network_precision_t* __restrict__ s_sample_r = network_rgb + sample_offset;
	const tcnn::network_precision_t* __restrict__ s_sample_g = s_sample_r + batch_size;
	const tcnn::network_precision_t* __restrict__ s_sample_b = s_sample_g + batch_size;

	float* __restrict__ s_dL_dweight = dL_dweight + sample_offset;
	float* __restrict__ s_dL_dcolor = dL_dcolor + sample_offset;

	const float dL_dR_r = dL_dR[idx + 0 * batch_size];
	const float dL_dR_g = dL_dR[idx + 1 * batch_size];
	const float dL_dR_b = dL_dR[idx + 2 * batch_size];
	const float dL_dR_a = dL_dR[idx + 3 * batch_size];

	for (int i = 0; i < n_samples; ++i) {
		const float sr = (float)s_sample_r[i];
		const float sg = (float)s_sample_g[i];
		const float sb = (float)s_sample_b[i];

		s_dL_dweight[i] = dL_dR_r * sr + dL_dR_g * sg + dL_dR_b * sb;
		s_dL_dcolor[i + 0 * batch_size] = dL_dR_r * s_weight[i];
		s_dL_dcolor[i + 1 * batch_size] = dL_dR_g * s_weight[i];
		s_dL_dcolor[i + 2 * batch_size] = dL_dR_b * s_weight[i];
	}
}

// RGBA to loss
__global__ void ray_rgba_to_loss_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const float* __restrict__ ray_rgba,
	const float* __restrict__ target_rgba,
	float* __restrict__ loss
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) {
		return;
	}

	const uint32_t r_idx = idx;
	const uint32_t g_idx = r_idx + batch_size;
	const uint32_t b_idx = g_idx + batch_size;
	const uint32_t a_idx = b_idx + batch_size;

	const float dr = target_rgba[r_idx] - ray_rgba[r_idx];
	const float dg = target_rgba[g_idx] - ray_rgba[g_idx];
	const float db = target_rgba[b_idx] - ray_rgba[b_idx];
	const float da = target_rgba[a_idx] - ray_rgba[a_idx];
	
	loss[r_idx] = (1.0f / (3.0f * (float)n_rays)) * (dr * dr);
	loss[g_idx] = (1.0f / (3.0f * (float)n_rays)) * (dg * dg);
	loss[b_idx] = (1.0f / (3.0f * (float)n_rays)) * (db * db);
	// loss[a_idx] = (1.0f / (4.0f * (float)n_rays)) * (da * da);
}

// dL/dR = (1/4) * (dL/dR_r + dL/dR_g + dL/dR_b + dL/dR_a)
__global__ void ray_rgba_to_loss_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ ray_steps,
	const uint32_t* __restrict__ ray_steps_cum,
	const float* __restrict__ dt_batch,
	const float* __restrict__ ray_rgba,
	const float* __restrict__ target_rgba,
	const float* __restrict__ loss, // output from forward pass
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

	const float dr = target_rgba[r_idx] - ray_rgba[r_idx];
	const float dg = target_rgba[g_idx] - ray_rgba[g_idx];
	const float db = target_rgba[b_idx] - ray_rgba[b_idx];
	// const float da = target_rgba[a_idx] - ray_rgba[a_idx];
	
	dL_dR[r_idx] = -(2.0f / (3.0f)) * dr;
	dL_dR[g_idx] = -(2.0f / (3.0f)) * dg;
	dL_dR[b_idx] = -(2.0f / (3.0f)) * db;
	// dL_dR[a_idx] = -(2.0f / (4.0f)) * da;
}

NRC_NAMESPACE_END
