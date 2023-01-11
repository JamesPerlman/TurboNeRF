#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/cascaded-occupancy-grid.cuh"

NRC_NAMESPACE_BEGIN

__global__ void generate_rays_pinhole_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const Camera* __restrict__ cam,
	float* __restrict__ ray_ori,
	float* __restrict__ ray_dir,
	float* __restrict__ ray_idir,
    uint32_t* __restrict__ ray_idx,
	const uint32_t start_idx = 0,
	const uint32_t end_idx = UINT32_MAX
);

__global__ void march_rays_and_generate_samples_kernel(
    const uint32_t n_rays,
	const uint32_t batch_size,
	const BoundingBox* bbox,
	const CascadedOccupancyGrid* occ_grid,
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
	float* __restrict__ sample_pos,
	float* __restrict__ sample_dt
);

__global__ void composite_samples_kernel(
    const uint32_t n_samples,
    const uint32_t batch_size,
    
    // read-only
    const tcnn::network_precision_t* __restrict__ network_sigma,
    const tcnn::network_precision_t* __restrict__ network_rgb,
    const float* __restrict__ sample_dt,
    const uint32_t* __restrict__ sample_idx,

    // read/write
    bool* __restrict__ ray_alive,
    bool* __restrict__ ray_active,
    float* __restrict__ output_rgba
);

NRC_NAMESPACE_END
