#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"

NRC_NAMESPACE_BEGIN

// TODO: These can be rewritten with cuBLAS - check out NerfAcc

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

	// output buffers
	float* __restrict__ ray_rgba, // per ray
	float* __restrict__ sample_trans, // per sample
	float* __restrict__ sample_alpha,
	float* __restrict__ sample_weight
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_rays) {
		return;
	}

	// Grab local references to global data
	const uint32_t n_samples = n_samples_per_ray[i];
	const uint32_t sample_offset = n_samples_cum[i] - n_samples;

	const tcnn::network_precision_t* __restrict__ s_r = network_rgb + sample_offset;
	const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
	const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;

	const tcnn::network_precision_t* __restrict__ s_sigma = network_sigma + sample_offset;

	const float* __restrict__ s_dt = sample_dt + sample_offset;

	float* __restrict__ s_trans = sample_trans + sample_offset;
	float* __restrict__ s_alpha = sample_alpha + sample_offset;
	float* __restrict__ s_weight = sample_weight + sample_offset;

	// Values to accumulate samples into
	float ray_r = 0.0f;
	float ray_g = 0.0f;
	float ray_b = 0.0f;
	float ray_a = 0.0f;

	float sigma_cumsum = 0.0f;

	// Accumulate samples
	for (int j = 0; j < n_samples; ++j) {
		// thank you NerfAcc (render_transmittance.cu - transmittance_from_sigma_forward_kernel)
		const float dt = s_dt[j];
		
		const float sigma_j = (float)s_sigma[j];
		const float sigma_j_dt = sigma_j * dt;

		const float alpha = 1.0f - expf(-sigma_j_dt);
		const float trans = expf(-sigma_cumsum);
		
		sigma_cumsum += sigma_j_dt;

		const float weight = alpha * trans;

		// accumulate the color
		ray_r += weight * (float)s_r[j];
		ray_g += weight * (float)s_g[j];
		ray_b += weight * (float)s_b[j];
		ray_a += weight;

		// save transmittance for gradient calculation
		s_trans[j] = trans;
		s_alpha[j] = alpha;
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

	// squared error sum per ray color component
	out_loss[i] = dr * dr + dg * dg + db * db + da * da;

	// local references to sample data
	const uint32_t n_samples = n_samples_per_ray[i];
	const uint32_t sample_offset_0 = n_samples_cum[i] - n_samples;
	const uint32_t sample_offset_1 = sample_offset_0 + batch_size;
	const uint32_t sample_offset_2 = sample_offset_1 + batch_size;
	const uint32_t sample_offset_3 = sample_offset_2 + batch_size;
	
	// Store pixel difference values for gradient calculation
	for (int j = 0; j < n_samples; ++j) {
		pixel_diffs[sample_offset_0 + j] = dr;
		pixel_diffs[sample_offset_1 + j] = dg;
		pixel_diffs[sample_offset_2 + j] = db;
		pixel_diffs[sample_offset_3 + j] = da;
	}
}

/**
 * Here we calculate the gradients dL/doutput with respect to the color output (per channel/sample) and density output
 */
__global__ void calculate_network_output_gradient(
	const uint32_t n_samples,
	const uint32_t batch_size, // aka data stride
	const float inv_2npix, // 1 / (2.0f * n_pixels)
	const tcnn::network_precision_t* __restrict__ sample_color,
	const float* __restrict__ pixel_diffs, // signed difference per color channel, like (ray_r - gt_r).  stored for easy access in this kernel
	const float* __restrict__ sample_dt, // per sample
	const float* __restrict__ sample_trans,
	const float* __restrict__ sample_alpha,
	const float* __restrict__ sample_weight,

	const float loss_scale,

	// output
	tcnn::network_precision_t* __restrict__ grad
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;
	const uint32_t i_offset_3 = i_offset_2 + batch_size;

	if (i >= n_samples) {
		grad[i_offset_0] = 0.0f;
		grad[i_offset_1] = 0.0f;
		grad[i_offset_2] = 0.0f;
		grad[i_offset_3] = 0.0f;

		return;
	}

	// local references to data

	const float dt = sample_dt[i];
	const float weight = sample_weight[i];
	const float alpha = sample_alpha[i];
	const float trans = sample_trans[i];

	const float dr = pixel_diffs[i_offset_0];
	const float dg = pixel_diffs[i_offset_1];
	const float db = pixel_diffs[i_offset_2];
	const float da = pixel_diffs[i_offset_3];

	const float sr = (float)sample_color[i_offset_0];
	const float sg = (float)sample_color[i_offset_1];
	const float sb = (float)sample_color[i_offset_2];

	// for gradient formula derivations, look at "/research/NeRF Loss Function Derivation.pdf"

	// rgb gradient
	grad[i_offset_0] = (tcnn::network_precision_t)(loss_scale * inv_2npix * dr * weight);
	grad[i_offset_1] = (tcnn::network_precision_t)(loss_scale * inv_2npix * dg * weight);
	grad[i_offset_2] = (tcnn::network_precision_t)(loss_scale * inv_2npix * db * weight);

	// sigma gradient
	grad[i_offset_3] = (tcnn::network_precision_t)(loss_scale * inv_2npix * (1.0f - 2.0f * alpha) * (dt * trans) * (dr * sr + dg * sg + db * sb + da));
}

/**
 * Normalization and inversion kernels
 * We need to normalize data for the neural network, and then convert the coordinates back to world space after the network has processed the data
 */

__global__ void normalize_network_input_kernel(
	const uint32_t batch_size,
	const float inv_bbox_size, // 1 / (2 * bbox_size)
	const float* __restrict__ in_sample_pos, // input positions are assumed to be in [-bbox_size, bbox_size]
	const float* __restrict__ in_sample_dir, // input dirs are assumed to be normalized
	const float* __restrict__ in_sample_dt,
	float* __restrict__ out_sample_pos, // output positions are transformed to [0, 1]
	float* __restrict__ out_sample_dir, // output dirs are transformed to [0, 1]
	float* __restrict__ out_sample_dt
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batch_size) {
		return;
	}

	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;

	out_sample_pos[i_offset_0] = tcnn::clamp(in_sample_pos[i_offset_0] * inv_bbox_size + 0.5f, 0.0f, 1.0f);
	out_sample_pos[i_offset_1] = tcnn::clamp(in_sample_pos[i_offset_1] * inv_bbox_size + 0.5f, 0.0f, 1.0f);
	out_sample_pos[i_offset_2] = tcnn::clamp(in_sample_pos[i_offset_2] * inv_bbox_size + 0.5f, 0.0f, 1.0f);

	out_sample_dir[i_offset_0] = in_sample_dir[i_offset_0] * 0.5f + 0.5f;
	out_sample_dir[i_offset_1] = in_sample_dir[i_offset_1] * 0.5f + 0.5f;
	out_sample_dir[i_offset_2] = in_sample_dir[i_offset_2] * 0.5f + 0.5f;

	if (in_sample_dt != nullptr) {
		out_sample_dt[i] = inv_bbox_size * in_sample_dt[i];
	}
}

NRC_NAMESPACE_END
