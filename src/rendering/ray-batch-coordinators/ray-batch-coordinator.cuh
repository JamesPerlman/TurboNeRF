#pragma once

#include "../../common.h"
#include "../../models/bounding-box.cuh"
#include "../../models/camera.cuh"
#include "../../models/ray.h"
#include "../../models/ray-batch.cuh"
#include "../../render-targets/render-target.cuh"

NRC_NAMESPACE_BEGIN

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
        const BoundingBox* bbox,
        RayBatch& ray_batch,
        const cudaStream_t& stream = 0
    ) = 0;

    virtual void copy_packed(
        const int& n_pixels,
        const int& stride,
        const int2& output_size,
        float* rgba_in,
        float* rgba_out,
        const cudaStream_t& stream = 0
    ) = 0;

    virtual ~RayBatchCoordinator() {};
};

/**
 * Each subclass of RayBatchCoordinator will be responsible for generating rays, and must call a kernel.
 * This device function abstracts some common functionality:
 * + invert ray directions
 * + set default ray properties
 * + write to ray buffers
 */
inline __device__ void fill_ray_buffers(
    const int& i,
    const int& stride,
    const Ray& global_ray,
    const BoundingBox* __restrict__ bbox,
    float* __restrict__ pos,
    float* __restrict__ dir,
    float* __restrict__ idir,
    float* __restrict__ t,
    int* __restrict__ index,
    bool* __restrict__ alive
) {
    int i_offset_0 = i;
    int i_offset_1 = i_offset_0 + stride;
    int i_offset_2 = i_offset_1 + stride;
    
	const float3 global_origin = global_ray.o;
	const float3 global_direction = global_ray.d;

	const float dir_x = global_direction.x;
	const float dir_y = global_direction.y;
	const float dir_z = global_direction.z;
	
	const float idir_x = 1.0f / dir_x;
	const float idir_y = 1.0f / dir_y;
	const float idir_z = 1.0f / dir_z;
	
    // make sure this ray intersects the bbox
	float _t;
	const bool intersects_bbox = bbox->get_ray_t_intersection(
		global_origin.x, global_origin.y, global_origin.z,
		dir_x, dir_y, dir_z,
		idir_x, idir_y, idir_z,
		_t
	);

    if (!intersects_bbox) {
        // no "alive" bit
        alive[i] = false;
        return;
    }

	pos[i_offset_0] = global_origin.x;
	pos[i_offset_1] = global_origin.y;
	pos[i_offset_2] = global_origin.z;
	
	dir[i_offset_0] = dir_x;
	dir[i_offset_1] = dir_y;
	dir[i_offset_2] = dir_z;

	idir[i_offset_0] = idir_x;
	idir[i_offset_1] = idir_y;
	idir[i_offset_2] = idir_z;

    // set t-value to a position just barely within the bbox
	t[i] = fmaxf(0.0f, _t + 1e-5f);
    index[i] = i;
	alive[i] = true;
};

NRC_NAMESPACE_END
