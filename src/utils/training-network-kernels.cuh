#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <crt/device_functions.h>

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"

NRC_NAMESPACE_BEGIN

/**
 * Apply exponential scaling to density network output
 * (log-space density!)
 */
template <typename Output>
__global__ void apply_exp_to_density_kernel(
	uint32_t batch_size,
	const tcnn::network_precision_t* __restrict__ input,
	Output* __restrict__ output
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < batch_size) {
		output[i] = (Output)__expf((float)input[i]);
	}
}

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
	const tcnn::network_precision_t* __restrict__ network_sigma,
	const tcnn::network_precision_t* __restrict__ network_color,
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

	// sigma
	const tcnn::network_precision_t* __restrict__ s_s = network_sigma + sample_offset;

	// rgb
	const tcnn::network_precision_t* __restrict__ s_r = network_color + sample_offset;
	const tcnn::network_precision_t* __restrict__ s_g = s_r + batch_size;
	const tcnn::network_precision_t* __restrict__ s_b = s_g + batch_size;

	const float* __restrict__ s_dt = sample_dt + sample_offset;

	float* __restrict__ s_trans = sample_trans + sample_offset;
	float* __restrict__ s_alpha = sample_alpha + sample_offset;
	float* __restrict__ s_weight = sample_weight + sample_offset;

	// Values to accumulate samples into
	float ray_r = 0.0f;
	float ray_g = 0.0f;
	float ray_b = 0.0f;
	float ray_a = 0.0f;

	float sigma_dt_sum = 0.0f;

	// Accumulate samples
	for (int j = 0; j < n_samples; ++j) {
		// thank you NerfAcc (render_transmittance.cu - transmittance_from_sigma_forward_kernel)
		const float dt = s_dt[j];
		
		// we need to apply the __expf post-activation to treat the density network output as being in logarithmic space.
		const float sigma_j = __expf((float)s_s[j]);
		const float sigma_j_dt = sigma_j * dt;

		const float trans = __expf(-sigma_dt_sum);
		const float alpha = 1.0f - __expf(-sigma_j_dt);
		const float weight = alpha * trans;
		
		sigma_dt_sum += sigma_j_dt;

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
	const float n_pixels,
	const tcnn::network_precision_t* __restrict__ network_sigma,
	const tcnn::network_precision_t* __restrict__ network_rgb,
	const uint32_t* __restrict__ n_samples_per_ray,
	const uint32_t* __restrict__ n_samples_cum,
	const float* __restrict__ pixel_diffs, // signed difference per color channel, like (ray_r - gt_r)
	const float* __restrict__ sample_dt, // per sample
	const float* __restrict__ sample_trans,
	const float* __restrict__ sample_alpha,
	const float* __restrict__ sample_weight,
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
	const float* alpha = sample_alpha + sample_offset;
	const float* trans = sample_trans + sample_offset;

	const float dr = pixel_diffs[i + 0 * batch_size];
	const float dg = pixel_diffs[i + 1 * batch_size];
	const float db = pixel_diffs[i + 2 * batch_size];
	const float da = pixel_diffs[i + 3 * batch_size];

	const tcnn::network_precision_t* sr = network_rgb + sample_offset;
	const tcnn::network_precision_t* sg = sr + batch_size;
	const tcnn::network_precision_t* sb = sg + batch_size;

	const tcnn::network_precision_t* sd = network_sigma + sample_offset;

	// for gradient formula derivations, look at "/research/NeRF Loss Function Derivation.pdf"

	float sum_tn_an_rn = 0.0f;
	float sum_tn_an_gn = 0.0f;
	float sum_tn_an_bn = 0.0f;
	float sum_tn_an = 0.0f;

	// decrementing loop
	for (int j = n_samples - 1; j >= 0; --j) {
		// We need a lot of variables...
		const float es_j = __expf(tcnn::clamp((float)sd[j], -15.0f, 15.0f));
		const float dt_j = dt[j];
		const float T_j = trans[j];
		const float a_j = alpha[j];
		const float w_j = weight[j];
		
		const float r_j = sr[j];
		const float g_j = sg[j];
		const float b_j = sb[j];

		//const float T_j_a_j = T_j * a_j;
		const float dt_j_es_j = dt_j * es_j;
		
		sum_tn_an_rn += w_j * r_j;
		sum_tn_an_gn += w_j * g_j;
		sum_tn_an_bn += w_j * b_j;
		sum_tn_an += w_j;

		grad_r[j] = (tcnn::network_precision_t)(loss_scale * (1.0f / (2.0f * n_pixels)) * dr * w_j);
		grad_g[j] = (tcnn::network_precision_t)(loss_scale * (1.0f / (2.0f * n_pixels)) * dg * w_j);
		grad_b[j] = (tcnn::network_precision_t)(loss_scale * (1.0f / (2.0f * n_pixels)) * db * w_j);
		// grad_s[j] = (tcnn::network_precision_t)(loss_scale * (1.0f / (2.0f * n_pixels)) * (1.0f - 2.0f * a_j) * (dt_j * T_j * es_j) * (dr_j * r_j + dg_j * g_j + db_j * b_j + da_j));

		const float k = T_j * (1.0f - a_j);
		const float dLr_dsj = dr * (k * r_j - sum_tn_an_rn);
		const float dLg_dsj = dg * (k * g_j - sum_tn_an_gn);
		const float dLb_dsj = db * (k * b_j - sum_tn_an_bn);
		const float dLa_dsj = da * (k - sum_tn_an);

		grad_s[j] = (tcnn::network_precision_t)(loss_scale * (1.0f / (2.0f * n_pixels)) * dt_j_es_j * (dLr_dsj + dLg_dsj + dLb_dsj + dLa_dsj));
	}
}

// modify dL/doutput of the density matrix
__global__ void modify_density_dL_doutput_kernel(
	const uint32_t batch_size,
	const tcnn::network_precision_t* __restrict__ sigma_grad,
	tcnn::network_precision_t* __restrict__ dL_doutput
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < batch_size) {
		dL_doutput[i] = sigma_grad[i];
	}
}

NRC_NAMESPACE_END
