#pragma once

#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"
#include "../utils/nerf-constants.cuh"
#include "common-network-kernels.cuh"

TURBO_NAMESPACE_BEGIN

__global__ void mipNeRF360_distortion_loss_forward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,

    // input buffers
    // per ray
    const uint32_t* __restrict__ ray_steps,
    const uint32_t* __restrict__ ray_offset,

    // per sample
    const tcnn::network_precision_t* __restrict__ sample_density_buf,
    const float* __restrict__ sample_m_norm_buf,
    const float* __restrict__ sample_dt_norm_buf,

    // output buffers 
    // per ray
    float* __restrict__ ray_dtw2_cs_buf,
    float* __restrict__ ray_w_cs_buf,
    float* __restrict__ ray_wm_cs_buf,
    float* __restrict__ ray_wm_w_cs1_cs_buf,
    float* __restrict__ ray_w_wm_cs1_cs_buf,
    float* __restrict__ ray_dist_loss_buf,

    // per sample
    float* __restrict__ sample_w_cs_buf,
    float* __restrict__ sample_wm_cs_buf
) {

    const uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) {
        return;
    }

    const uint32_t n_samples = ray_steps[ray_idx];

    if (n_samples == 0) {
        ray_dist_loss_buf[ray_idx] = 0.0f;
        ray_w_cs_buf[ray_idx] = 0.0f;
        ray_wm_cs_buf[ray_idx] = 0.0f;
        ray_dtw2_cs_buf[ray_idx] = 0.0f;
        ray_wm_w_cs1_cs_buf[ray_idx] = 0.0f;
        ray_w_wm_cs1_cs_buf[ray_idx] = 0.0f;
        return;
    }

    const uint32_t sample_offset = ray_offset[ray_idx];

    const tcnn::network_precision_t* density = sample_density_buf + sample_offset;
    const float* dt = sample_dt_norm_buf + sample_offset;
    const float* m = sample_m_norm_buf + sample_offset;

    float* w_cs = sample_w_cs_buf + sample_offset;
    float* wm_cs = sample_wm_cs_buf + sample_offset;

    float cumsum_w = 0.0f;
    float cumsum_wm = 0.0f;
    float cumsum_dtw2 = 0.0f;
    float cumsum_wm_w_cs1 = 0.0f;
    float cumsum_w_wm_cs1 = 0.0f;

    float trans = 1.0f;
    float prev_w_cs = 0.0f;
    float prev_wm_cs = 0.0f;

    float loss_bi_0 = 0.0f;
    float loss_bi_1 = 0.0f;
    float loss_uni = 0.0f;

    for (uint32_t i = 0; i < n_samples; ++i) {
        const float dt_i = dt[i];
        const float sigma_i = density_to_sigma(density[i]);
        const float alpha_i = sigma_to_alpha(sigma_i, dt_i);
        const float trans_i = trans;
        const float weight_i = alpha_i * trans_i;
        const float m_i = m[i];
        const float wm_i = weight_i * m_i;

        const float wm_w_cs_1_i = wm_i * prev_w_cs;
        const float w_wm_cs_1_i = weight_i * prev_wm_cs;
        const float dtw2_i = dt_i * weight_i * weight_i;

        cumsum_w += weight_i;
        cumsum_wm += wm_i;
        cumsum_dtw2 += dtw2_i;
        cumsum_wm_w_cs1 += wm_w_cs_1_i;
        cumsum_w_wm_cs1 += w_wm_cs_1_i;

        w_cs[i] = cumsum_w;
        wm_cs[i] = cumsum_wm;

        loss_bi_0 += wm_w_cs_1_i;
        loss_bi_1 += w_wm_cs_1_i;
        loss_uni += dtw2_i;

        prev_w_cs = cumsum_w;
        prev_wm_cs = cumsum_wm;

        trans *= (1.0f - alpha_i);
        
        if (trans < NeRFConstants::min_transmittance) {
            break;
        }
    }

    const float k = NeRFConstants::mipNeRF360_distortion_loss_lambda / (float)n_samples;
    ray_dist_loss_buf[ray_idx] = k * ((1.0 / 3.0) * loss_uni + 2.0 * (loss_bi_0 - loss_bi_1));
    ray_w_cs_buf[ray_idx] = cumsum_w;
    ray_wm_cs_buf[ray_idx] = cumsum_wm;
    ray_dtw2_cs_buf[ray_idx] = cumsum_dtw2;
    ray_wm_w_cs1_cs_buf[ray_idx] = cumsum_wm_w_cs1;
    ray_w_wm_cs1_cs_buf[ray_idx] = cumsum_w_wm_cs1;
}


__global__ void mipNeRF360_distortion_loss_backward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,

    // input buffers
    // per ray
    const uint32_t* __restrict__ ray_steps,
    const uint32_t* __restrict__ ray_offset,
    // per sample
    const tcnn::network_precision_t* __restrict__ sample_density_buf,

    // output buffers
    // per ray
    const float* __restrict__ ray_dtw2_cs_buf,
    const float* __restrict__ ray_w_cs_buf,
    const float* __restrict__ ray_wm_cs_buf,
    const float* __restrict__ ray_wm_w_cs1_cs_buf,
    const float* __restrict__ ray_w_wm_cs1_cs_buf,
    const float* __restrict__ ray_dist_loss_buf,
    
    // per sample
    const float* __restrict__ sample_m_norm_buf,
    const float* __restrict__ sample_dt_norm_buf,
    const float* __restrict__ sample_w_cs_buf,
    const float* __restrict__ sample_wm_cs_buf,

    // output buffer
    float* __restrict__ sample_dloss_ddensity_buf
) {

    const uint32_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= n_rays) {
        return;
    }

    const uint32_t n_samples = ray_steps[ray_idx];
    
    if (n_samples == 0) {
        return;
    }

    const uint32_t sample_offset = ray_offset[ray_idx];

    const tcnn::network_precision_t* density = sample_density_buf + sample_offset;
    const float* dt = sample_dt_norm_buf + sample_offset;
    const float* m = sample_m_norm_buf + sample_offset;
    const float* w_cs = sample_w_cs_buf + sample_offset;
    const float* wm_cs = sample_wm_cs_buf + sample_offset;

    float* sample_dloss_ddensity = sample_dloss_ddensity_buf + sample_offset;

    float cumsum_w = ray_w_cs_buf[ray_idx];
    float cumsum_wm = ray_wm_cs_buf[ray_idx];
    float cumsum_dtw2 = ray_dtw2_cs_buf[ray_idx];
    float cumsum_wm_w_cs1 = ray_wm_w_cs1_cs_buf[ray_idx];
    float cumsum_w_wm_cs1 = ray_w_wm_cs1_cs_buf[ray_idx];

    float trans = 1.0f;
    float prev_w_cs = 0.0f;
    float prev_wm_cs = 0.0f;

    const float n_samples_f = (float)n_samples;

    const float k = NeRFConstants::mipNeRF360_distortion_loss_lambda / n_samples_f;

    for (uint32_t i = 0; i < n_samples; ++i) {
        const float dt_i = dt[i];
        const float sigma_i = density_to_sigma(density[i]);
        const float alpha_i = sigma_to_alpha(sigma_i, dt_i);
        const float trans_i = trans;
        const float weight_i = alpha_i * trans_i;
        const float m_i = m[i];
        const float wm_i = weight_i * m_i;
        const float w_cs_i = w_cs[i];
        const float wm_cs_i = wm_cs[i];

        const float dalpha_dsigma_i = dt_i * (1.0f - alpha_i);
        const float dw_dsigma_i = trans_i * dalpha_dsigma_i;
        const float dwm_dsigma_i = m_i * dw_dsigma_i;
        const float t_w_i = trans_i - weight_i;

        cumsum_w -= weight_i;
        cumsum_wm -= wm_i;
        cumsum_dtw2 -= dt_i * weight_i * weight_i;
        cumsum_wm_w_cs1 -= wm_i * prev_w_cs;
        cumsum_w_wm_cs1 -= weight_i * prev_wm_cs;
        
        const float dloss_bi_0_dsigma = dwm_dsigma_i * prev_w_cs + dt_i * (cumsum_wm * (w_cs_i + t_w_i) - 2.0f * cumsum_wm_w_cs1);
        const float dloss_bi_1_dsigma = dw_dsigma_i * prev_wm_cs + dt_i * (cumsum_w * (m_i * t_w_i + wm_cs_i) - 2.0f * cumsum_w_wm_cs1);
        const float dloss_uni_dsigma = 2.0f * dt_i * (dt_i * weight_i * t_w_i - cumsum_dtw2);

        const float dloss_dsigma_i = k * (2.0f * (dloss_bi_0_dsigma - dloss_bi_1_dsigma) + (1.0f / 3.0f) * dloss_uni_dsigma);
        const float dsigma_ddensity_i = sigma_i;

        sample_dloss_ddensity[i] = dloss_dsigma_i * dsigma_ddensity_i;

        prev_w_cs = w_cs_i;
        prev_wm_cs = wm_cs_i;

        trans *= (1.0f - alpha_i);

        if (trans < NeRFConstants::min_transmittance) {
            break;
        }
    }
}

// sigma to ray color
inline __device__ void density_to_ray_rgba_forward(
    const uint32_t& n_samples,
    const tcnn::network_precision_t* __restrict__ sample_r,
    const tcnn::network_precision_t* __restrict__ sample_g,
    const tcnn::network_precision_t* __restrict__ sample_b,
    const tcnn::network_precision_t* __restrict__ sample_density,
    const float* __restrict__ sample_dt,
    float& ray_r, float& ray_g, float& ray_b, float& ray_a
) {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;

    float trans = 1.0f;

    for (int i = 0; i < n_samples; ++i) {
        const float alpha = density_to_alpha(sample_density[i], sample_dt[i]);
        
        const float weight = trans * alpha;

        r += weight * (float)sample_r[i];
        g += weight * (float)sample_g[i];
        b += weight * (float)sample_b[i];
        a += weight;

        trans *= (1.0f - alpha);

        if (trans < NeRFConstants::min_transmittance) {
            break;
        }
    }

    ray_r = r;
    ray_g = g;
    ray_b = b;
    ray_a = a;
}

// sigma to ray color backward
// gets dL/ddensity = dL/dray_color * dray_color/dsigma * dsigma/ddensity

inline __device__ void density_to_ray_rgba_backward(
    const uint32_t& idx,
    const uint32_t& batch_size,
    const uint32_t& n_samples,
    const float& dL_dcolor_scale,
    const float& rand_r, const float& rand_g, const float& rand_b,
    const float& ray_r, const float& ray_g, const float& ray_b, const float& ray_a,
    const float& dL_dR_r, const float& dL_dR_g, const float& dL_dR_b,
    const tcnn::network_precision_t* __restrict__ sample_r,
    const tcnn::network_precision_t* __restrict__ sample_g,
    const tcnn::network_precision_t* __restrict__ sample_b,
    const tcnn::network_precision_t* __restrict__ sample_density,
    const float* __restrict__ sample_dt,
    float* __restrict__ sample_dL_ddensity,
    tcnn::network_precision_t* __restrict__ sample_dL_dcolor_r,
    tcnn::network_precision_t* __restrict__ sample_dL_dcolor_g,
    tcnn::network_precision_t* __restrict__ sample_dL_dcolor_b
) {

    const float dr = ray_r - rand_r;
    const float dg = ray_g - rand_g;
    const float db = ray_b - rand_b;

    float cumsum_r = ray_r;
    float cumsum_g = ray_g;
    float cumsum_b = ray_b;
    float cumsum_a = ray_a;

    float trans = 1.0f;
    for (int i = 0; i < n_samples; ++i) {
        const float r = sample_r[i];
        const float g = sample_g[i];
        const float b = sample_b[i];

        const float dt = sample_dt[i];
        const float sigma = density_to_sigma(sample_density[i]);
        const float alpha = sigma_to_alpha(sigma, dt);
        
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

        const float dL_dsigma = dL_dR_r * dRr_dsigma + dL_dR_g * dRg_dsigma + dL_dR_b * dRb_dsigma;
        const float dsigma_ddensity = sigma;

        sample_dL_ddensity[i] = dL_dsigma * dsigma_ddensity;

        const float dR_dcolor = weight * ray_a;
        
        sample_dL_dcolor_r[i] = (tcnn::network_precision_t)(dL_dcolor_scale * dL_dR_r * dR_dcolor);
        sample_dL_dcolor_g[i] = (tcnn::network_precision_t)(dL_dcolor_scale * dL_dR_g * dR_dcolor);
        sample_dL_dcolor_b[i] = (tcnn::network_precision_t)(dL_dcolor_scale * dL_dR_b * dR_dcolor);
        
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
inline __device__ void ray_rgba_to_loss_forward(
    const float& dr, const float& dg, const float& db, const float& da,
    float& loss_r, float& loss_g, float& loss_b, float& loss_a
) {

    loss_r = smooth_l1_loss_forward(dr);
    loss_g = smooth_l1_loss_forward(dg);
    loss_b = smooth_l1_loss_forward(db);
    loss_a = smooth_l1_loss_forward(da);
}

// dL/dR = (1/3) * (dL/dR_r + dL/dR_g + dL/dR_b)
inline __device__ void ray_rgba_to_loss_backward(
    const float& inv_3nrays,
    const float& dr, const float& dg, const float& db,
    float& dL_dR_r, float& dL_dR_g, float& dL_dR_b
) {
    dL_dR_r = inv_3nrays * smooth_l1_loss_backward(dr);
    dL_dR_g = inv_3nrays * smooth_l1_loss_backward(dg);
    dL_dR_b = inv_3nrays * smooth_l1_loss_backward(db);
}

__global__ void fused_reconstruction_forward_backward_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const float inv_3nrays,
    const float grad_scale,
    const uint32_t* __restrict__ ray_steps,
    const uint32_t* __restrict__ ray_offset,
    const float* __restrict__ random_rgb,
    const float* __restrict__ target_rgba,
    const tcnn::network_precision_t* __restrict__ density_buf,
    const tcnn::network_precision_t* __restrict__ color_buf,
    const float* __restrict__ dt_buf,
    float* __restrict__ loss_buf,
    float* __restrict__ dL_ddensity_buf,
    tcnn::network_precision_t* __restrict__ dL_dcolor_buf
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    const uint32_t i_offset_0 = i;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;
    const uint32_t i_offset_3 = i_offset_2 + batch_size;

    const uint32_t sample_offset_0 = ray_offset[i];
    const uint32_t sample_offset_1 = sample_offset_0 + batch_size;
    const uint32_t sample_offset_2 = sample_offset_1 + batch_size;
    
    const uint32_t n_samples = ray_steps[i];

    // some references to sample buffers
    const tcnn::network_precision_t* __restrict__ sample_r = color_buf + sample_offset_0;
    const tcnn::network_precision_t* __restrict__ sample_g = color_buf + sample_offset_1;
    const tcnn::network_precision_t* __restrict__ sample_b = color_buf + sample_offset_2;
    const tcnn::network_precision_t* __restrict__ sample_density = density_buf + sample_offset_0;
    const float* __restrict__ sample_dt = dt_buf + sample_offset_0;

    // First, calculate ray color
    float ray_r, ray_g, ray_b, ray_a;
    density_to_ray_rgba_forward(
        n_samples,
        sample_r, sample_g, sample_b,
        sample_density,
        sample_dt,
        ray_r, ray_g, ray_b, ray_a
    );

    // https://github.com/cheind/pure-torch-ngp/blob/develop/torchngp/training.py#L301-L314
    // mixing random colors with predicted and ground truth colors encourages the network to learn empty space
    // THANK YOU CHEIND

    const float gt_r = target_rgba[i_offset_0];
    const float gt_g = target_rgba[i_offset_1];
    const float gt_b = target_rgba[i_offset_2];
    const float gt_a = target_rgba[i_offset_3];

    const float rand_r = random_rgb[i_offset_0];
    const float rand_g = random_rgb[i_offset_1];
    const float rand_b = random_rgb[i_offset_2];

    const float gt_a_comp = 1.0f - gt_a;
    const float gt_r_comp = gt_r * gt_a + rand_r * gt_a_comp;
    const float gt_g_comp = gt_g * gt_a + rand_g * gt_a_comp;
    const float gt_b_comp = gt_b * gt_a + rand_b * gt_a_comp;

    const float ray_a_comp = 1.0f - ray_a;
    const float ray_r_comp = ray_r * ray_a + rand_r * ray_a_comp;
    const float ray_g_comp = ray_g * ray_a + rand_g * ray_a_comp;
    const float ray_b_comp = ray_b * ray_a + rand_b * ray_a_comp;

    const float dr = ray_r_comp - gt_r_comp;
    const float dg = ray_g_comp - gt_g_comp;
    const float db = ray_b_comp - gt_b_comp;
    const float da = ray_a_comp - gt_a_comp;

    // Next, calculate loss
    float loss_r, loss_g, loss_b, loss_a;
    ray_rgba_to_loss_forward(
        dr, dg, db, da,
        loss_r, loss_g, loss_b, loss_a
    );

    loss_buf[i_offset_0] = loss_r;
    loss_buf[i_offset_1] = loss_g;
    loss_buf[i_offset_2] = loss_b;
    loss_buf[i_offset_3] = loss_a;

    // Now we can backpropagate

    float dL_dR_r, dL_dR_g, dL_dR_b;
    ray_rgba_to_loss_backward(
        inv_3nrays,
        dr, dg, db,
        dL_dR_r, dL_dR_g, dL_dR_b
    );

    float* __restrict__ sample_dL_ddensity = dL_ddensity_buf + sample_offset_0;
    tcnn::network_precision_t* __restrict__ sample_dL_dcolor_r = dL_dcolor_buf + sample_offset_0;
    tcnn::network_precision_t* __restrict__ sample_dL_dcolor_g = dL_dcolor_buf + sample_offset_1;
    tcnn::network_precision_t* __restrict__ sample_dL_dcolor_b = dL_dcolor_buf + sample_offset_2; 

    density_to_ray_rgba_backward(
        i,
        batch_size,
        n_samples,
        grad_scale,
        rand_r, rand_g, rand_b,
        ray_r, ray_g, ray_b, ray_a,
        dL_dR_r, dL_dR_g, dL_dR_b,
        sample_r, sample_g, sample_b,
        sample_density,
        sample_dt,
        sample_dL_ddensity,
        sample_dL_dcolor_r, sample_dL_dcolor_g, sample_dL_dcolor_b
    );
}

TURBO_NAMESPACE_END
