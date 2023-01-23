#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <crt/device_functions.h>

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

	sigma[i] = (T)__expf(tcnn::clamp((float)density[i] - 1.0f, -10.0f, 10.0f));
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
	
	dL_ddensity[i] = dL_dsigma[i] * __expf(tcnn::clamp((float)density[i] - 1.0f, -10.0f, 10.0f));
}

/**
 * THESE ARE DELIBERATELY UNDER-OPTIMIZED BECAUSE I AM LEARNING HOW TO BACKPROPAGATE
 */

/**
 * Transmittance from activated sigma
 */

// calculates transmittance from sigma
__global__ void sigma_to_transmittance_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
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
	float* __restrict__ s_trans = transmittance + sample_offset;

	for (int i = 0; i < n_samples; ++i) {
		float sigma_cumsum = 0.0f;
		for (int j = 0; j < i; ++j) {
			sigma_cumsum += s_sigma[j] * s_dt[j];
		}
		s_trans[i] = __expf(-sigma_cumsum);
	}
}


// calculates dL/dsigma = dL/dtransmittance * dtransmittance/dsigma
__global__ void sigma_to_transmittance_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	const float* __restrict__ transmittance, // forward output
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
	const float* __restrict__ s_dL_dtrans = dL_dtransmittance + sample_offset;
	float* __restrict__ s_dL_dsigma = dL_dsigma + sample_offset;

    for (int i = 0; i < n_samples; ++i) {
        float cumsum_outer = 0.0f;
        for (int j = i + 1; j < n_samples; ++j) {
            float cumsum_inner = 0.0;
            for (int k = 0; k < j; ++k) {
                cumsum_inner += sigma[k] * dt[k];
            }
            cumsum_outer += __expf(-cumsum_inner) * s_dL_dtrans[j]; //?
        }
        s_dL_dsigma[i] = -dt[i] * cumsum_outer;
    }
}

// calculates weight from sigma, per sample
__global__ void sigma_to_weight_forward_kernel(
	uint32_t n_rays,
	uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	float* __restrict__ weight
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) return;

	// offsets
	const uint32_t n_samples = n_samples_per_ray[idx];
	const uint32_t sample_offset = n_samples_cum[idx] - n_samples;

	// local references to sample data
	const float* __restrict__ s_dt = dt + sample_offset;
	const float* __restrict__ s_sigma = sigma + sample_offset;
	float* __restrict__ s_weight = weight + sample_offset;

	for (int i = 0; i < n_samples; ++i) {
		float sigma_cumsum = 0.0f;
		for (int j = 0; j < i; ++j) {
			sigma_cumsum += s_sigma[j] * s_dt[j];
		}
		const float trans = __expf(-sigma_cumsum);
		const float alpha = 1.0f - __expf(-s_sigma[i] * s_dt[i]);
		s_weight[i] = trans * alpha;
	}
}

// calculates dL/dsigma = dL/dweight * dweight/dsigma
__global__ void sigma_to_weight_backward_kernel(
	uint32_t n_rays,
	uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	const float* __restrict__ weight, // forward output
	const float* __restrict__ dL_dweight,
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
	const float* __restrict__ s_weight = weight + sample_offset;
	const float* __restrict__ s_dL_dweight = dL_dweight + sample_offset;
	float* __restrict__ s_dL_dsigma = dL_dsigma + sample_offset;

	for (int i = 0; i < n_samples; ++i) {
		float sigma_cumsum = 0.0f;
		for (int j = 0; j < i; ++j) {
			sigma_cumsum += s_sigma[j] * s_dt[j];
		}
		const float dweight_dsigma = s_dt[i] * __expf(-sigma_cumsum)  * __expf(-s_sigma[i] * s_dt[i]);
		s_dL_dsigma[i] = s_dL_dweight[i] * dweight_dsigma; 
	}
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

		s_dL_dweight[i] = dL_dR_a + dL_dR_r * sr + dL_dR_g * sg + dL_dR_b * sb;
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
	
	loss[r_idx] = (1.0f / (4.0f * (float)n_rays)) * (dr * dr);
	loss[g_idx] = (1.0f / (4.0f * (float)n_rays)) * (dg * dg);
	loss[b_idx] = (1.0f / (4.0f * (float)n_rays)) * (db * db);
	loss[a_idx] = (1.0f / (4.0f * (float)n_rays)) * (da * da);
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
	const float da = target_rgba[a_idx] - ray_rgba[a_idx];
	
	dL_dR[r_idx] = -(2.0f / (4.0f)) * dr;
	dL_dR[g_idx] = -(2.0f / (4.0f)) * dg;
	dL_dR[b_idx] = -(2.0f / (4.0f)) * db;
	dL_dR[a_idx] = -(2.0f / (4.0f)) * da;
}

// calculates dL/dR across all channels
__global__ void ray_channel_to_loss_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const float* __restrict__ predicted_channel,
	const float* __restrict__ groundtruth_channel,
	const float* __restrict__ loss, // result from forward pass
	float* __restrict__ dL_dR
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) {
		return;
	}

	float diff = groundtruth_channel[idx] - predicted_channel[idx];
	dL_dR[idx] = (2.0f / (float)n_rays) * diff;
}


__global__ void sigma_to_alpha_forward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	float* __restrict__ alpha
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

	for (int i = 0; i < n_samples; ++i) {
		s_alpha[i] = 1.0f - __expf(-s_sigma[i] * s_dt[i]);
	}
}

__global__ void alpha_from_sigma_backward_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ dt,
	const float* __restrict__ sigma,
	const float* __restrict__ alpha,
	const float* __restrict__ alpha_grad,
	float* __restrict__ sigma_grad
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
	const float* __restrict__ s_alpha_grad = alpha_grad + sample_offset;
	float* __restrict__ s_sigma_grad = sigma_grad + sample_offset;

	for (int i = 0; i < n_samples; ++i) {
		s_sigma_grad[i] = -s_dt[i] * s_alpha[i] * s_alpha_grad[i];
	}
}


// TODO: These can be rewritten with cuBLAS - check out NerfAcc

/**
 * Accumulate the sample colors by ray.
 * This is differentiable wrt inputs network_color, sample_sigma, sample_dt
 */

__global__ void accumulate_ray_colors_from_samples_kernel(
	uint32_t n_rays, // n_rays is not necessarily the batch size here
	uint32_t batch_size,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,

	// these input buffers organized based on the cumulative steps
	const tcnn::network_precision_t* __restrict__ network_sigma,
	const tcnn::network_precision_t* __restrict__ network_color,
	const float* __restrict__ sample_sigma,
	const float* __restrict__ sample_dt,

	// output buffers
	float* __restrict__ ray_rgba, // per ray
	float* __restrict__ sample_trans, // per sample
	float* __restrict__ sample_weight
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_rays) {
		return;
	}

	// Grab local references to global data
	const uint32_t n_samples = n_samples_per_ray[i];
	const uint32_t sample_offset = n_samples_cum[i] - n_samples;

	// sigma
	const float* __restrict__ s_s = sample_sigma + sample_offset;
	const tcnn::network_precision_t* __restrict__ net_sig = network_sigma + sample_offset;

	// rgb
	const tcnn::network_precision_t* __restrict__ s_r = network_color + sample_offset;
	const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
	const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;

	const float* __restrict__ s_dt = sample_dt + sample_offset;

	float* __restrict__ s_trans = sample_trans + sample_offset;
	float* __restrict__ s_weight = sample_weight + sample_offset;

	// Values to accumulate samples into
	float ray_r = 0.0f;
	float ray_g = 0.0f;
	float ray_b = 0.0f;
	float ray_a = 0.0f;

	float sigma_dt_sum = 0.0f;

	// Accumulate samples
	for (int j = 0; j < n_samples; ++j) {
		const float dt = s_dt[j];
		
		const float sigma_j = __expf(tcnn::clamp((float)net_sig[j] - 1.0f, -15.0f, 15.0f));
		const float sigma_j_dt = sigma_j * dt;

		const float trans = __expf(-sigma_dt_sum);
		sigma_dt_sum += sigma_j_dt;

		const float alpha = 1.0f - __expf(-sigma_j_dt);
		const float weight = alpha * trans;

		// accumulate the color
		ray_r += weight * (float)s_r[j];
		ray_g += weight * (float)s_g[j];
		ray_b += weight * (float)s_b[j];
		ray_a += weight;

		// save transmittance and weight for gradient calculation
		s_trans[j] = trans;
		s_weight[j] = weight;
	}
	
	// write out the accumulated ray color
	ray_rgba[i + 0 * batch_size] = ray_r;
	ray_rgba[i + 1 * batch_size] = ray_g;
	ray_rgba[i + 2 * batch_size] = ray_b;
	ray_rgba[i + 3 * batch_size] = ray_a;
}

// Calculate the loss (squared sum of errors per ray)
__global__ void calculate_sse_loss_per_ray_kernel(
	uint32_t n_pixels,
	uint32_t batch_size, // aka data stride
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ ray_rgba, // this is the accumulated ray color
	const float* __restrict__ target_rgba, // the ground-truth pixel colors for each ray
	const float* __restrict__ sample_sigma,
    // const float loss_scale,
	float* __restrict__ out_loss, // per ray
	float* __restrict__ pixel_diffs // difference in ray color vs ground truth, stored per sample index
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
	const float dr = ray_rgba[i_offset_0] - target_rgba[i_offset_0];
	const float dg = ray_rgba[i_offset_1] - target_rgba[i_offset_1];
	const float db = ray_rgba[i_offset_2] - target_rgba[i_offset_2];
	const float da = ray_rgba[i_offset_3] - target_rgba[i_offset_3];

	// squared sum of errors per ray color component
	out_loss[i] = dr * dr + dg * dg + db * db + da * da;
	
	// Store pixel difference values for gradient calculation
	pixel_diffs[i_offset_0] = dr;
	pixel_diffs[i_offset_1] = dg;
	pixel_diffs[i_offset_2] = db;
	pixel_diffs[i_offset_3] = da;
}

/**
 * Here we calculate the gradients dL/doutput with respect to the color output (per channel/sample) and density output
 */
__global__ void calculate_network_output_gradient(
	const uint32_t n_rays,
	const uint32_t batch_size, // aka data stride
	const float inv_2npix,
	const float n_pix,
	const tcnn::network_precision_t* __restrict__ network_sigma,
	const tcnn::network_precision_t* __restrict__ network_rgb,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ pixel_diffs, // signed difference per color channel, like (ray_r - gt_r)
	const float* __restrict__ sample_dt, // per sample
	const float* __restrict__ sample_trans,
	const float* __restrict__ sample_weight,
	const float* __restrict__ sample_sigma,
	const float loss_scale,

	// output
	tcnn::network_precision_t* __restrict__ sigma_grad,
	tcnn::network_precision_t* __restrict__ color_grad
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_rays) {
		return;
	}
	
	const uint32_t n_samples = n_samples_per_ray[i];
	const uint32_t sample_offset = n_samples_cum[i] - n_samples;

	tcnn::network_precision_t* grad_r = color_grad + sample_offset;
	tcnn::network_precision_t* grad_g = grad_r + batch_size;
	tcnn::network_precision_t* grad_b = grad_g + batch_size;

	tcnn::network_precision_t* grad_s = sigma_grad + sample_offset;

	// local references to sample buffers
	const float* dt = sample_dt + sample_offset;
	const float* weight = sample_weight + sample_offset;
	const float* trans = sample_trans + sample_offset;
	const float* sigma = sample_sigma + sample_offset;

	const float dr = pixel_diffs[i + 0 * batch_size];
	const float dg = pixel_diffs[i + 1 * batch_size];
	const float db = pixel_diffs[i + 2 * batch_size];
	const float da = pixel_diffs[i + 3 * batch_size];

	const tcnn::network_precision_t* sr = network_rgb + sample_offset;
	const tcnn::network_precision_t* sg = sr + batch_size;
	const tcnn::network_precision_t* sb = sg + batch_size;

	// for gradient formula derivations, look at "/research/NeRF Loss Function Derivation.pdf"
	const tcnn::network_precision_t* net_sig = network_sigma + sample_offset;

	// decrementing loop
	for (int j = 0; j < n_samples; ++j) {
		// We need a lot of variables...
		const float T_j = trans[j];
		// stop raymarching if transmittance is too small?
		const float es_j = __expf(tcnn::clamp((float)net_sig[j] - 1.0f, -15.0f, 15.0f));
		const float dt_j = dt[j];
		const float w_j = weight[j];
		
		const float r_j = sr[j];
		const float g_j = sg[j];
		const float b_j = sb[j];

		// RGB gradients are pretty simple...

		grad_r[j] = (tcnn::network_precision_t)(loss_scale * inv_2npix * dr * w_j);
		grad_g[j] = (tcnn::network_precision_t)(loss_scale * inv_2npix * dg * w_j);
		grad_b[j] = (tcnn::network_precision_t)(loss_scale * inv_2npix * db * w_j);

		// Sigma gradients are a bit more complicated...

		// const float dLr_dsj = 4.0f * inv_2npix * dr * (dt_j * es_j) * ((T_j - w_j) * r_j - sum_wn_rn);
		float sum_wn_rn = 0.0f;
		float sum_wn_gn = 0.0f;
		float sum_wn_bn = 0.0f;
		float sum_wn = 0.0f;

		for (int n = j + 1; n < n_samples; ++n) {
			sum_wn_rn += weight[n] * (float)sr[n];
			sum_wn_gn += weight[n] * (float)sg[n];
			sum_wn_bn += weight[n] * (float)sb[n];
			sum_wn += weight[n];
		}

		const float k = T_j - w_j;

		const float dLr_dsj = 2.0f / n_pix * dr * (dt_j * es_j) * (r_j * (T_j - w_j) - sum_wn_rn);
		const float dLg_dsj = 2.0f / n_pix * dg * (dt_j * es_j) * (g_j * (T_j - w_j) - sum_wn_gn);
		const float dLb_dsj = 2.0f / n_pix * db * (dt_j * es_j) * (b_j * (T_j - w_j) - sum_wn_bn);
		// const float dLa_dsj = 4.0f * inv_2npix * da * (dt_j * es_j) * ((T_j - w_j) - sum_wn);
		const float dLa_dsj = 2.0f / n_pix * da * (dt_j * es_j) * ((T_j - w_j) - sum_wn);

		grad_s[j] = (tcnn::network_precision_t)(loss_scale * 0.25f * (dLr_dsj + dLg_dsj + dLb_dsj));
	}
}

NRC_NAMESPACE_END
