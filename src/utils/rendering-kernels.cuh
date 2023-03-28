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

__global__ void march_rays_to_first_occupied_cell_kernel(
    const uint32_t n_rays,
	const uint32_t batch_size,
	const OccupancyGrid* grid,
	const BoundingBox* bbox,
	const float dt_min,
	const float dt_max,
	const float cone_angle,
	
	// input buffers (read-only)
	const float* __restrict__ ray_dir,
	const float* __restrict__ ray_idir,

    // dual-use buffers (read/write)
    bool* __restrict__ ray_alive,
	float* __restrict__ ray_ori,
    float* __restrict__ ray_t,
	float* __restrict__ ray_t_max,

	// output buffers (write-only)
	float* __restrict__ network_pos,
	float* __restrict__ network_dir,
	float* __restrict__ network_dt
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
	const Camera* __restrict__ cameras,
	const stbi_uc* __restrict__ train_img_data,
	const float* __restrict__ ray_ori,
	const float* __restrict__ ray_dir,
	float* __restrict__ ray_t_max,
	float* __restrict__ out_rgba_buf
);

__global__ void march_rays_and_generate_network_inputs_kernel(
    const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t n_steps_max,
	const uint32_t network_stride,
	const OccupancyGrid* grid,
	const BoundingBox* bbox,
	const float inv_aabb_size,
	const float dt_min,
	const float dt_max,
	const float cone_angle,
	
	// input buffers (read-only)
	const float* __restrict__ ray_ori,
	const float* __restrict__ ray_dir,
	const float* __restrict__ ray_idir,
	const float* __restrict__ ray_t_max,

    // dual-use buffers (read/write)
    bool* __restrict__ ray_alive,
    bool* __restrict__ ray_active,
    float* __restrict__ ray_t,

	// output buffers (write-only)
	uint32_t* __restrict__ n_ray_steps,
	float* __restrict__ network_pos,
	float* __restrict__ network_dir,
	float* __restrict__ network_dt
);

__global__ void compact_rays_kernel(
    const int n_compacted_rays,
	const int batch_size,
    const int* __restrict__ indices,

	// input buffers (read-only)
	const int* __restrict__ in_idx, // this is the ray-pixel index
	const bool* __restrict__ in_active,
	const float* __restrict__ in_t,
	const float* __restrict__ in_t_max,
	const float* __restrict__ in_origin,
	const float* __restrict__ in_dir,
	const float* __restrict__ in_idir,
	const float* __restrict__ in_trans,

	// compacted output buffers (write-only)
	int* __restrict__ out_idx,
	bool* __restrict__ out_active,
	float* __restrict__ out_t,
	float* __restrict__ out_t_max,
	float* __restrict__ out_origin,
	float* __restrict__ out_dir,
	float* __restrict__ out_idir,
	float* __restrict__ out_trans
);

__global__ void composite_samples_kernel(
	const uint32_t n_rays,
	const uint32_t network_stride,
	const uint32_t output_stride,

    // read-only
	const bool* __restrict__ ray_active,
	const uint32_t* __restrict__ n_ray_steps,
    const int* __restrict__ ray_idx,
	const tcnn::network_precision_t* __restrict__ network_output,
	const float* __restrict__ sample_alpha,

    // read/write
    bool* __restrict__ ray_alive,
	float* __restrict__ ray_trans,
    float* __restrict__ output_rgba
);

__global__ void alpha_composite_kernel(
    const uint32_t n_pixels,
    const uint32_t img_stride,
    const float* rgba_fg,
    const float* rgba_bg,
	float* rgba_out
);

TURBO_NAMESPACE_END
