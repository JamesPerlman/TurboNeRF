#pragma once

#include "../../common.h"
#include "../../models/bounding-box.cuh"
#include "../../models/camera.cuh"
#include "../../models/ray.h"
#include "../../models/ray-batch.cuh"
#include "../../render-targets/render-target.cuh"

TURBO_NAMESPACE_BEGIN

/**
 * A ray batch coordinator is responsible for two main things:
 * 1. Generate a batch of rays from a camera
 * 2. Copy the results to a render target
 * 
 * These currently only apply to rendering batches, not training batches.
 */

class RayBatchCoordinator {
public:
    virtual void generate_rays(
        const Camera* camera,
        RayBatch& ray_batch,
        const cudaStream_t& stream = 0
    ) = 0;

    virtual void copy_packed(
        const int& n_rays,
        const int2& output_size,
        const int& output_stride,
        float* rgba_in,
        float* rgba_out,
        const cudaStream_t& stream = 0
    ) = 0;

    virtual ~RayBatchCoordinator() {};
};

/**
 * Each subclass of RayBatchCoordinator will be responsible for generating rays, and must call a kernel.
 * This device function abstracts some common functionality:
 * + set default ray properties
 * + write to ray buffers
 */
inline __device__ void fill_ray_buffers(
    const int& i,
    const int& stride,
    const Camera* __restrict__ cam,
    const int& pix_idx_x,
    const int& pix_idx_y,
    float* __restrict__ pos,
    float* __restrict__ dir,
    float* __restrict__ t_max,
    int* __restrict__ index,
    bool* __restrict__ alive
) {
    int i_offset_0 = i;
    int i_offset_1 = i_offset_0 + stride;
    int i_offset_2 = i_offset_1 + stride;

    const Ray global_ray = cam->global_ray_at_pixel_xy((float)pix_idx_x, (float)pix_idx_y);
    
	const float3 global_ori = global_ray.o;
	const float3 global_dir = global_ray.d;

	const float dir_x = global_dir.x;
	const float dir_y = global_dir.y;
	const float dir_z = global_dir.z;
	
    // assign ray properties

	pos[i_offset_0] = global_ori.x;
	pos[i_offset_1] = global_ori.y;
	pos[i_offset_2] = global_ori.z;
	
	dir[i_offset_0] = dir_x;
	dir[i_offset_1] = dir_y;
	dir[i_offset_2] = dir_z;

    t_max[i] = cam->far - cam->near;
    index[i] = i;
	alive[i] = true;
};

TURBO_NAMESPACE_END
