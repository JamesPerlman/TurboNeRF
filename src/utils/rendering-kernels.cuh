#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../core/occupancy-grid.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "common-network-kernels.cuh"

NRC_NAMESPACE_BEGIN

__global__ void generate_rays_pinhole_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const BoundingBox* __restrict__ bbox,
	const Camera* __restrict__ cam,
	const uint32_t start_idx,
	float* __restrict__ ray_ori,
	float* __restrict__ ray_dir,
	float* __restrict__ ray_idir,
	float* __restrict__ ray_t,
	float* __restrict__ ray_trans,
    uint32_t* __restrict__ ray_idx,
	bool* __restrict__ ray_alive,
	bool* __restrict__ ray_active
);

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

	// output buffers (write-only)
	float* __restrict__ network_pos,
	float* __restrict__ network_dir,
	float* __restrict__ network_dt
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
	const float* __restrict__ in_origin,
	const float* __restrict__ in_dir,
	const float* __restrict__ in_idir,
	const float* __restrict__ in_trans,

	// compacted output buffers (write-only)
	int* __restrict__ out_idx,
	bool* __restrict__ out_active,
	float* __restrict__ out_t,
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

NRC_NAMESPACE_END
