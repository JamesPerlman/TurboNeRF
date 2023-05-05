#pragma once

#include <cuda_runtime.h>
#include <stbi/stb_image.h>
#include <stdint.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../core/occupancy-grid.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "common-network-kernels.cuh"

TURBO_NAMESPACE_BEGIN

__global__ void prepare_for_linear_raymarching_kernel(
    const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t n_nerfs,
	const OccupancyGrid* grids,
	const BoundingBox* bboxes,
	const Transform4f* transforms,
	const float dt_min,
	const float cone_angle,
	
	// input buffers (read-only)
	const float* __restrict__ ray_ori,
	const float* __restrict__ ray_dir,

    // dual-use buffers (read/write)
    bool* __restrict__ ray_alive,
	float* __restrict__ ray_tmax,

	// output buffers (write-only)
	uint32_t* __restrict__ intersectors,
	bool* __restrict__ nerf_ray_active,
    float* __restrict__ nerf_ray_t,
	float* __restrict__ nerf_tmax
);

__global__ void draw_training_img_clipping_planes_and_assign_t_max_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t out_rgba_stride,
	const uint32_t n_cameras,
	const int2 training_img_dims,
	const uint32_t n_pix_per_training_img,
	const bool show_near_planes,
	const bool show_far_planes,
	const Transform4f* __restrict__ nerf_transform,
	const Camera* __restrict__ cameras,
	const stbi_uc* __restrict__ train_img_data,
	const float* __restrict__ ray_ori,
	const float* __restrict__ ray_dir,
	float* __restrict__ ray_t_max,
	float* __restrict__ out_rgba_buf
);

__global__ void march_rays_and_generate_network_inputs_kernel(
    const uint32_t n_rays,
	const uint32_t n_nerfs,
	const uint32_t batch_size,
	const uint32_t network_batch,
	const uint32_t n_samples_per_step,
	const uint32_t n_steps_max,
	const OccupancyGrid* grids,
	const BoundingBox* bboxes,
	const Transform4f* transforms,
	const float dt_min,
	const float cone_angle,
	
	// input buffers (read-only)
	const float* __restrict__ ray_ori,
	const float* __restrict__ ray_dir,
	const float* __restrict__ ray_tmax,
	const float* __restrict__ nerf_tmax,
	const uint32_t* __restrict__ intersectors,

    // dual-use buffers (read/write)
    bool* __restrict__ ray_alive,
    bool* __restrict__ nerf_ray_active,
    float* __restrict__ nerf_ray_t,

	// output buffers (write-only)
	int* __restrict__ n_steps_total,
	int* __restrict__ sample_nerf_id,
	float* __restrict__ network_pos,
	float* __restrict__ network_dir,
	float* __restrict__ network_dt
);

__global__ void compact_network_inputs_kernel(
	const uint32_t n_compacted_samples,
	const uint32_t old_batch_size,
	const uint32_t new_batch_size,
	const int* __restrict__ indices,

	// input buffers (read-only)
	const float* __restrict__ in_network_pos,
	const float* __restrict__ in_network_dir,

	// output buffers (write-only)
	float* __restrict__ out_network_pos,
	float* __restrict__ out_network_dir
);

__global__ void expand_network_outputs_kernel(
	const uint32_t n_compacted_samples,
	const uint32_t old_batch_size,
	const uint32_t new_batch_size,
	const int* __restrict__ indices,

	// input buffers (read-only)
	const tcnn::network_precision_t* __restrict__ in_network_rgb,
	const tcnn::network_precision_t* __restrict__ in_network_density,

	// output buffers (write-only)
	tcnn::network_precision_t* __restrict__ out_network_rgb,
	tcnn::network_precision_t* __restrict__ out_network_density
);

__global__ void composite_samples_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t network_stride,
	const uint32_t output_stride,
	const uint32_t n_samples_per_step,
	const uint32_t n_steps_max,
	const uint32_t n_nerfs,

    // read-only
    const int* __restrict__ ray_idx,
	const float* __restrict__ ray_dt,
	const tcnn::network_precision_t* __restrict__ network_rgb,
	const tcnn::network_precision_t* __restrict__ network_density,
	const int* __restrict__ n_steps_total,

    // read/write
	float* __restrict__ ray_trans,
    float* __restrict__ output_rgba,

	// write-only
	bool* __restrict__ ray_alive
);

__global__ void compact_rays_kernel(
    const int n_compacted_rays,
	const int n_nerfs,
	const int batch_size,
    const int* __restrict__ indices,

	// input buffers (read-only)
	const int* __restrict__ in_idx, // this is the ray-pixel index
	const bool* __restrict__ in_nerf_ray_active,
	const float* __restrict__ in_nerf_ray_t,
	const float* __restrict__ in_nerf_ray_tmax,
	const uint32_t* __restrict__ in_intersectors,
	const float* __restrict__ in_ray_tmax,
	const float* __restrict__ in_ori,
	const float* __restrict__ in_dir,
	const float* __restrict__ in_trans,

	// compacted output buffers (write-only)
	int* __restrict__ out_idx,
	bool* __restrict__ out_nerf_ray_active,
	float* __restrict__ out_nerf_ray_t,
	float* __restrict__ out_nerf_ray_tmax,
	uint32_t* __restrict__ out_intersectors,
	float* __restrict__ out_ray_tmax,
	float* __restrict__ out_ori,
	float* __restrict__ out_dir,
	float* __restrict__ out_trans
);

__global__ void alpha_composite_kernel(
    const uint32_t n_pixels,
    const uint32_t img_stride,
    const float* rgba_fg,
    const float* rgba_bg,
	float* rgba_out
);

TURBO_NAMESPACE_END
