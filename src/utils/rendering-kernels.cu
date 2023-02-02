
#include <device_launch_parameters.h>

#include "rendering-kernels.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../utils/color-utils.cuh"
#include "../utils/nerf-constants.cuh"

using namespace tcnn;

NRC_NAMESPACE_BEGIN

// TODO: move this into a Camera utility kernel file
// init_rays CUDA kernel
__global__ void generate_rays_pinhole_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const BoundingBox* __restrict__ bbox,
	const Camera* __restrict__ cam,
	float* __restrict__ ray_ori,
	float* __restrict__ ray_dir,
	float* __restrict__ ray_idir,
	float* __restrict__ ray_t,
	float* __restrict__ ray_trans,
    uint32_t* __restrict__ ray_idx,
	bool* __restrict__ ray_alive,
	bool* __restrict__ ray_active,
	const uint32_t start_idx
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_rays) {
		return;
	}

	uint32_t idx = start_idx + i;
	
	uint32_t x = idx % cam->pixel_dims.x;
	uint32_t y = idx / cam->pixel_dims.x;

	Ray local_ray = cam->local_ray_at_pixel_xy(x, y);

    float3 global_origin = cam->transform * local_ray.o;
	float3 global_direction = cam->transform * local_ray.d - cam->transform.get_translation();

	// normalize ray directions
	const float n = rnorm3df(global_direction.x, global_direction.y, global_direction.z);

	const float dir_x = n * global_direction.x;
	const float dir_y = n * global_direction.y;
	const float dir_z = n * global_direction.z;

	const float idir_x = 1.0f / dir_x;
	const float idir_y = 1.0f / dir_y;
	const float idir_z = 1.0f / dir_z;

    // save data to buffers
	uint32_t i_offset_0 = i;
	uint32_t i_offset_1 = i_offset_0 + batch_size;
	uint32_t i_offset_2 = i_offset_1 + batch_size;

	ray_ori[i_offset_0] = global_origin.x;
	ray_ori[i_offset_1] = global_origin.y;
	ray_ori[i_offset_2] = global_origin.z;

	ray_dir[i_offset_0] = dir_x;
	ray_dir[i_offset_1] = dir_y;
	ray_dir[i_offset_2] = dir_z;

	ray_idir[i_offset_0] = idir_x;
	ray_idir[i_offset_1] = idir_y;
	ray_idir[i_offset_2] = idir_z;

	float t;
	const bool intersects_bbox = bbox->get_ray_t_intersection(
		global_origin.x, global_origin.y, global_origin.z,
		dir_x, dir_y, dir_z,
		idir_x, idir_y, idir_z,
		t
	);

	ray_t[i] = intersects_bbox ? fmaxf(0.0f, t + 1e-5f) : 0.0f;

	ray_alive[i] = intersects_bbox;

    ray_idx[i] = idx;

	ray_active[i] = true;

	ray_trans[i] = 1.0f;
}

__global__ void march_rays_and_generate_network_inputs_kernel(
    const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t n_steps_max,
	const uint32_t network_stride,
	const CascadedOccupancyGrid* occ_grid,
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
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= n_rays) return;

    // check if ray has terminated or is currently inactive
    if (!ray_alive[i] || !ray_active[i]) return;

	// References to input buffers
	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;

	const float o_x = ray_ori[i_offset_0];
	const float o_y = ray_ori[i_offset_1];
	const float o_z = ray_ori[i_offset_2];

	const float d_x = ray_dir[i_offset_0];
	const float d_y = ray_dir[i_offset_1];
	const float d_z = ray_dir[i_offset_2];
	
	const float id_x = ray_idir[i_offset_0];
	const float id_y = ray_idir[i_offset_1];
	const float id_z = ray_idir[i_offset_2];

	// Perform raymarching
	
	float t = ray_t[i];
	uint32_t n_steps = 0;

	while (n_steps < n_steps_max) {
		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bbox->contains(x, y, z)) {
			ray_alive[i] = false;
			break;
		}

		const float dt = occ_grid->get_dt(t, cone_angle, dt_min, dt_max);
		const int grid_level = occ_grid->get_grid_level_at(x, y, z, dt);

		if (occ_grid->is_occupied_at(grid_level, x, y, z)) {

			const uint32_t step_offset_0 = n_steps * n_rays + i; // coalesced!
			const uint32_t step_offset_1 = step_offset_0 + network_stride;
			const uint32_t step_offset_2 = step_offset_1 + network_stride;

			network_pos[step_offset_0] = x * inv_aabb_size + 0.5f;
			network_pos[step_offset_1] = y * inv_aabb_size + 0.5f;
			network_pos[step_offset_2] = z * inv_aabb_size + 0.5f;

			network_dir[step_offset_0] = 0.5f * d_x + 0.5f;
			network_dir[step_offset_1] = 0.5f * d_y + 0.5f;
			network_dir[step_offset_2] = 0.5f * d_z + 0.5f;

			network_dt[step_offset_0] = dt * inv_aabb_size;

			t += dt;

			++n_steps;
		} else {
			// otherwise we need to find the next occupied cell
			t += occ_grid->get_dt_to_next_voxel(
				x, y, z,
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				dt_min,
				grid_level
			);
		}
	}

	ray_t[i] = t;
	n_ray_steps[i] = n_steps;
}

// ray compaction
__global__ void compact_rays_kernel(
    const int n_compacted_rays,
	const int batch_size,
    const int* __restrict__ indices,

	// input buffers (read-only)
	const uint32_t* __restrict__ in_idx, // this is the ray-pixel index
	const bool* __restrict__ in_active,
	const float* __restrict__ in_t,
	const float* __restrict__ in_origin,
	const float* __restrict__ in_dir,
	const float* __restrict__ in_idir,
	const float* __restrict__ in_trans,

	// compacted output buffers (write-only)
	uint32_t* __restrict__ out_idx,
	bool* __restrict__ out_active,
	float* __restrict__ out_t,
	float* __restrict__ out_origin,
	float* __restrict__ out_dir,
	float* __restrict__ out_idir,
	float* __restrict__ out_trans
) {
    // compacted index is the index to write to
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= n_compacted_rays) return;

	// expanded index is the index to read from
	const int e_idx = indices[c_idx];
	
	// 1-component buffers
	out_idx[c_idx]		= in_idx[e_idx];
	out_active[c_idx]	= in_active[e_idx];
	out_t[c_idx]		= in_t[e_idx];
	out_trans[c_idx]	= in_trans[e_idx];

	// local references to pointer offsets
	const int c_offset_0 = c_idx;
	const int c_offset_1 = c_offset_0 + batch_size;
	const int c_offset_2 = c_offset_1 + batch_size;

	const int e_offset_0 = e_idx;
	const int e_offset_1 = e_offset_0 + batch_size;
	const int e_offset_2 = e_offset_1 + batch_size;

	// 3-component buffers
	out_origin[c_offset_0] = in_origin[e_offset_0];
	out_origin[c_offset_1] = in_origin[e_offset_1];
	out_origin[c_offset_2] = in_origin[e_offset_2];

	out_dir[c_offset_0] = in_dir[e_offset_0];
	out_dir[c_offset_1] = in_dir[e_offset_1];
	out_dir[c_offset_2] = in_dir[e_offset_2];

	out_idir[c_offset_0] = in_idir[e_offset_0];
	out_idir[c_offset_1] = in_idir[e_offset_1];
	out_idir[c_offset_2] = in_idir[e_offset_2];

}

// alpha compositing kernel, composites the latest samples into the output image
__global__ void composite_samples_kernel(
	const uint32_t n_rays,
	const uint32_t network_stride,
	const uint32_t output_stride,

    // read-only
	const bool* __restrict__ ray_active,
	const uint32_t* __restrict__ n_ray_steps,
    const uint32_t* __restrict__ ray_idx,
    const float* __restrict__ sample_dt,
    const network_precision_t* __restrict__ network_output,

    // read/write
    bool* __restrict__ ray_alive,
	float* __restrict__ ray_trans,
    float* __restrict__ output_rgba
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

    // check if ray has terminated or is currently inactive
    if (!ray_alive[idx] || !ray_active[idx]) return;

	// pixel indices
	const uint32_t idx_offset_0 = ray_idx[idx];
	const uint32_t idx_offset_1 = idx_offset_0 + output_stride;
	const uint32_t idx_offset_2 = idx_offset_1 + output_stride;
	const uint32_t idx_offset_3 = idx_offset_2 + output_stride;

	// accumulated pixel colors
	float out_r = output_rgba[idx_offset_0];
	float out_g = output_rgba[idx_offset_1];
	float out_b = output_rgba[idx_offset_2];
	float out_a = output_rgba[idx_offset_3];

	// iterate through steps
	const uint32_t n_steps = n_ray_steps[idx];

	float trans = ray_trans[idx];

	for (int i = 0; i < n_steps; ++i) {
		
		const uint32_t step_offset_0 = i * n_rays + idx;
		const uint32_t step_offset_1 = step_offset_0 + network_stride;
		const uint32_t step_offset_2 = step_offset_1 + network_stride;
		const uint32_t step_offset_3 = step_offset_2 + network_stride;

		// sample properties
		const float dt = sample_dt[step_offset_0];
		const float sigma = (float)network_output[step_offset_3];
		const float alpha = 1.0f - __expf(-sigma * dt);
		const float weight = alpha * trans;

		// composite the same way we do accumulation during training
		const float s_r = __linear_to_srgb((float)network_output[step_offset_0]);
		const float s_g = __linear_to_srgb((float)network_output[step_offset_1]);
		const float s_b = __linear_to_srgb((float)network_output[step_offset_2]);

		out_r += weight * s_r;
		out_g += weight * s_g;
		out_b += weight * s_b;
		out_a += weight;

		// update and threshold transmittance
		trans *= 1.0f - alpha;

		if (trans <= NeRFConstants::min_transmittance) {
			ray_alive[idx] = false;
			break;
		}
	}

	// maxed out alpha = normalize color components and terminate ray
	// if (out_a >= 1.0f) {
	// 	ray_alive[idx] = false;
	
	// 	out_r /= out_a;
	// 	out_g /= out_a;
	// 	out_b /= out_a;
	// 	out_a = 1.0f;
	// }

	output_rgba[idx_offset_0] = out_r;
	output_rgba[idx_offset_1] = out_g;
	output_rgba[idx_offset_2] = out_b;
	output_rgba[idx_offset_3] = out_a;

	ray_trans[idx] = trans;

}

NRC_NAMESPACE_END
