
#include <device_launch_parameters.h>

#include "rendering-kernels.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/cascaded-occupancy-grid.cuh"

using namespace tcnn;

NRC_NAMESPACE_BEGIN

// TODO: move this into a Camera utility kernel file
// init_rays CUDA kernel
__global__ void generate_rays_pinhole_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const Camera* __restrict__ cam,
	float* __restrict__ ray_ori,
	float* __restrict__ ray_dir,
	float* __restrict__ ray_idir,
    uint32_t* __restrict__ ray_idx,
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
	const float n = rsqrtf(l2_squared_norm(global_direction));

	const float ray_dx = n * global_direction.x;
	const float ray_dy = n * global_direction.y;
	const float ray_dz = n * global_direction.z;

    // save data to buffers
	uint32_t i_offset_0 = i;
	uint32_t i_offset_1 = i_offset_0 + batch_size;
	uint32_t i_offset_2 = i_offset_1 + batch_size;

	ray_ori[i_offset_0] = global_origin.x;
	ray_ori[i_offset_1] = global_origin.y;
	ray_ori[i_offset_2] = global_origin.z;

	ray_dir[i_offset_0] = ray_dx;
	ray_dir[i_offset_1] = ray_dy;
	ray_dir[i_offset_2] = ray_dz;

	ray_idir[i_offset_0] = 1.0f / ray_dx;
	ray_idir[i_offset_1] = 1.0f / ray_dy;
	ray_idir[i_offset_2] = 1.0f / ray_dz;

    ray_idx[i] = idx;
}

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
	uint32_t n_steps_taken = 0;

	while (true) {

		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bbox->contains(x, y, z)) {
            ray_alive[i] = false;
			sample_pos[i_offset_0] = x;
			sample_pos[i_offset_1] = y;
			sample_pos[i_offset_2] = z;
			break;
		}

		int grid_level = occ_grid->get_grid_level_at(x, y, z, dt_min);

		if (occ_grid->is_occupied_at(grid_level, x, y, z)) {
			// if grid is occupied here, march forward by a calculated dt
			float dt = occ_grid->get_dt(t, cone_angle, dt_min, dt_max);

			sample_pos[i_offset_0] = x;
			sample_pos[i_offset_1] = y;
			sample_pos[i_offset_2] = z;

			// march t forward
			ray_t[i] = t + dt;
            sample_dt[i] = dt;

            // for now, we only march samples once.
            break;
		} else {
			// otherwise we need to find the next occupied cell
			t = occ_grid->get_t_advanced_to_next_voxel(
				bbox->pos_to_unit_x(x), bbox->pos_to_unit_y(y), bbox->pos_to_unit_z(z),
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				t, dt_min
			);
		}
	}
}

// ray compaction
__global__ void compact_rays_kernel(
    const int n_compacted_elements,
	const int batch_size,
    const int* __restrict__ indices,

	// input buffers (read-only)
	const uint32_t* __restrict__ in_idx, // this is the ray-pixel index
	const bool* __restrict__ in_active,
	const float* __restrict__ in_t,
	const float* __restrict__ in_origin,
	const float* __restrict__ in_dir,
	const float* __restrict__ in_idir,

	// compacted output buffers (write-only)
	uint32_t* __restrict__ out_idx,
	bool* __restrict__ out_active,
	float* __restrict__ out_t,
	float* __restrict__ out_origin,
	float* __restrict__ out_dir,
	float* __restrict__ out_idir
) {
    // compacted index is the index to write to
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= n_compacted_elements) return;

	// expanded index is the index to read from
	const int e_idx = indices[c_idx];
	
	// 1-component buffers
	out_idx[c_idx] = in_idx[e_idx];
	out_active[c_idx] = in_active[e_idx];
	out_t[c_idx] = in_t[e_idx];

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
    const uint32_t n_samples,
	const uint32_t batch_size,
	const uint32_t output_stride,
    
    // read-only
    const network_precision_t* __restrict__ network_sigma,
    const network_precision_t* __restrict__ network_rgb,
    const float* __restrict__ sample_dt,
    const uint32_t* __restrict__ sample_idx,
	const bool* __restrict__ ray_active,

    // read/write
    bool* __restrict__ ray_alive,
    float* __restrict__ output_rgba
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_samples) return;

    // check if ray has terminated or is currently inactive
    if (!ray_alive[i] || !ray_active[i]) return;

    // grab local references to global memory
    const uint32_t i_offset_0 = i;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;
    const uint32_t i_offset_3 = i_offset_2 + batch_size;
    
    const uint32_t idx_offset_0 = sample_idx[i];
    const uint32_t idx_offset_1 = idx_offset_0 + output_stride;
    const uint32_t idx_offset_2 = idx_offset_1 + output_stride;
    const uint32_t idx_offset_3 = idx_offset_2 + output_stride;

    // sample colors
    const float s_r = network_rgb[i_offset_0];
    const float s_g = network_rgb[i_offset_1];
    const float s_b = network_rgb[i_offset_2];

    // alpha
    const float s_a = 1.0f - exp(-(float)network_sigma[i_offset_0] * sample_dt[i]);
    

    // pixel colors
    const float p_r = output_rgba[idx_offset_0];
    const float p_g = output_rgba[idx_offset_1];
    const float p_b = output_rgba[idx_offset_2];

    // alpha
    const float p_a = output_rgba[idx_offset_3];

    // transmittance
    const float p_t = 1.0f - p_a;

    // alpha compositing
    // new samples are composited behind current pixels
    // aka new sample is the background, current pixel is the foreground

    const float output_a = p_a + s_a * p_t;

    output_rgba[idx_offset_0] = p_r * p_a + s_r * s_a * p_t;
    output_rgba[idx_offset_1] = p_g * p_a + s_g * s_a * p_t;
    output_rgba[idx_offset_2] = p_b * p_a + s_b * s_a * p_t;
    output_rgba[idx_offset_3] = output_a;

    // terminate ray if alpha >= 1.0
    if (output_a >= 1.0f) {
        ray_alive[i] = false;
    }
}

NRC_NAMESPACE_END
