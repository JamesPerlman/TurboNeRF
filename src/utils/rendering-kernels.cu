
#include <device_launch_parameters.h>
#include <stbi/stb_image.h>

#include "rendering-kernels.cuh"
#include "../core/occupancy-grid.cuh"
#include "../math/geometric-intersections.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../utils/color-utils.cuh"
#include "../utils/nerf-constants.cuh"

using namespace tcnn;

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
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= n_rays) return;

    // check if ray has terminated or is currently inactive
    if (!ray_alive[i]) return;

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

    // make sure this ray intersects the bbox
	float _t;
	const bool intersects_bbox = bbox->get_ray_t_intersection(
		o_x, o_y, o_z,
		d_x, d_y, d_z,
		id_x, id_y, id_z,
		_t
	);

	if (!intersects_bbox) {
		ray_alive[i] = false;
		return;
	}

    float t = fmaxf(ray_t[i], _t + 1e-5f);
	float t_max = ray_t_max[i];

	// Perform raymarching
	
	while (t < t_max) {
		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bbox->contains(x, y, z)) {
			break;
		}

		const float dt = grid->get_dt(t, cone_angle, dt_min, dt_max);
		const int grid_level = grid->get_grid_level_at(x, y, z, dt);

		if (grid->is_occupied_at(grid_level, x, y, z)) {
			ray_ori[i_offset_0] = x;
			ray_ori[i_offset_1] = y;
			ray_ori[i_offset_2] = z;
			ray_t[i] = 0.0f;
			ray_t_max[i] = t_max - t;

			return;
		} else {
			// otherwise we need to find the next occupied cell
			t += grid->get_dt_to_next_voxel(
				x, y, z,
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				dt_min,
				grid_level
			);
		}
	};

	// if we get here, then the ray has terminated
	ray_alive[i] = false;
}

inline __device__ bool intersects_plane_at_distance(
	const Camera& cam,
	const float& dist,
	const Transform4f& c2w,
	const Transform4f& w2c,
	const float3& c2w_xyz,
	const float3& ray_o,
	const float3& ray_d,
	float2& uv,
	float& t
) {
	// hacky but less operations
	const float3 v{ c2w.m02 * dist, c2w.m12 * dist, c2w.m22 * dist };
	const float3 plane_center = v + c2w_xyz;
	const float3 plane_normal = normalized(v);
	const float2 near_size{
		dist * cam.resolution_f.x / cam.focal_length.x,
		dist * cam.resolution_f.y / cam.focal_length.y
	};

	return ray_plane_intersection(
		ray_o,
		ray_d,
		plane_center,
		plane_normal,
		near_size,
		w2c,
		uv,
		t
	);
}

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
) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n_rays) return;

	const uint32_t i_offset_0 = idx;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;

	const float3 ray_o = make_float3(
		ray_ori[i_offset_0],
		ray_ori[i_offset_1],
		ray_ori[i_offset_2]
	);

	const float3 ray_d = make_float3(
		ray_dir[i_offset_0],
		ray_dir[i_offset_1],
		ray_dir[i_offset_2]
	);

	float t_min = FLT_MAX;
	float2 t_min_uv;
	int t_min_cam_idx = -1;

	// we are looking for the minimum t-value of any plane that intersects this ray
	for (int i = 0; i < n_cameras; ++i) {
		const Camera cam = cameras[i];
		
		if (!cam.show_image_planes)
			continue;

		const Transform4f c2w = cam.transform;
		const Transform4f w2c = c2w.inverse();
		const float3 c2w_xyz = c2w.get_translation();
		
		if (show_near_planes) {
			float t_near;
			float2 uv_near;

			bool intersects_near = intersects_plane_at_distance(
				cam,
				cam.near,
				c2w,
				w2c,  
				c2w_xyz,
				ray_o,
				ray_d,
				uv_near,
				t_near
			);

			if (intersects_near && t_near < t_min) {
				t_min = t_near;
				t_min_uv = uv_near;
				t_min_cam_idx = i;
				continue;
			}
		}

		if (show_far_planes) {
			// need to check the far plane now
			float t_far;
			float2 uv_far;

			bool intersects_far = intersects_plane_at_distance(
				cam,
				cam.far,
				c2w,
				w2c,
				c2w_xyz,
				ray_o,
				ray_d,
				uv_far,
				t_far
			);

			if (intersects_far && t_far < t_min) {
				t_min = t_far;
				t_min_uv = uv_far;
				t_min_cam_idx = i;
				continue;
			}
		}
	}

	// output pixel index
	const int out_idx_offset_0 = idx;
	const int out_idx_offset_1 = out_idx_offset_0 + (int)out_rgba_stride;
	const int out_idx_offset_2 = out_idx_offset_1 + (int)out_rgba_stride;
	const int out_idx_offset_3 = out_idx_offset_2 + (int)out_rgba_stride;
	
	// did we intersect anything?
	if (t_min_cam_idx > -1) {
		int2 pix_xy{
			(int)(t_min_uv.x * (float)training_img_dims.x),
			(int)(t_min_uv.y * (float)training_img_dims.y)
		};

		// clamp to the image bounds
		pix_xy.x = clamp(pix_xy.x, 0, training_img_dims.x - 1);
		pix_xy.y = clamp(pix_xy.y, 0, training_img_dims.y - 1);

		// get the pixel index
		const int train_pix_offset = n_pix_per_training_img * t_min_cam_idx;
		const int train_pix_idx = pix_xy.y * training_img_dims.x + pix_xy.x;

		const stbi_uc* train_rgba = train_img_data + 4 * (train_pix_offset + train_pix_idx);
		
		// write the pixel
		out_rgba_buf[out_idx_offset_0] = __srgb_to_linear((float)train_rgba[0] / 255.0f);
		out_rgba_buf[out_idx_offset_1] = __srgb_to_linear((float)train_rgba[1] / 255.0f);
		out_rgba_buf[out_idx_offset_2] = __srgb_to_linear((float)train_rgba[2] / 255.0f);
		out_rgba_buf[out_idx_offset_3] = (float)train_rgba[3] / 255.0f;

		// set t_max
		ray_t_max[idx] = t_min;
	} else {
		// clear output pixel
		out_rgba_buf[out_idx_offset_0] = 0.0f;
		out_rgba_buf[out_idx_offset_1] = 0.0f;
		out_rgba_buf[out_idx_offset_2] = 0.0f;
		out_rgba_buf[out_idx_offset_3] = 0.0f;
	}
}

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
	float t_max = ray_t_max[i];

	uint32_t n_steps = 0;

	while (n_steps < n_steps_max && t < t_max) {
		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bbox->contains(x, y, z)) {
			break;
		}

		const float dt = grid->get_dt(t, cone_angle, dt_min, dt_max);
		const int grid_level = grid->get_grid_level_at(x, y, z, dt);

		if (grid->is_occupied_at(grid_level, x, y, z)) {

			const uint32_t step_offset_0 = n_steps * n_rays + i;
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
			t += grid->get_dt_to_next_voxel(
				x, y, z,
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				dt_min,
				grid_level
			);
		}
	}
	
	n_ray_steps[i] = n_steps;
	ray_alive[i] = n_steps > 0;
	ray_t[i] = t;
}

// ray compaction
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
	out_t_max[c_idx]	= in_t_max[e_idx];
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
    const int* __restrict__ ray_idx,
	const network_precision_t* __restrict__ network_output,
    const float* __restrict__ sample_alpha,

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
	const int idx_offset_0 = ray_idx[idx];
	const int idx_offset_1 = idx_offset_0 + (int)output_stride;
	const int idx_offset_2 = idx_offset_1 + (int)output_stride;
	const int idx_offset_3 = idx_offset_2 + (int)output_stride;

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

		// sample properties
		const float alpha = sample_alpha[step_offset_0];
		const float weight = alpha * trans;

		// composite the same way we do accumulation during training
		const float s_r = (float)network_output[step_offset_0];
		const float s_g = (float)network_output[step_offset_1];
		const float s_b = (float)network_output[step_offset_2];

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

	output_rgba[idx_offset_0] = out_r;
	output_rgba[idx_offset_1] = out_g;
	output_rgba[idx_offset_2] = out_b;
	output_rgba[idx_offset_3] = out_a;

	ray_trans[idx] = trans;
}

// Thank you Copilot + GPT-4!
__global__ void alpha_composite_kernel(
    const uint32_t n_pixels,
    const uint32_t img_stride,
    const float* rgba_fg,
    const float* rgba_bg,
	float* rgba_out
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_pixels) return;

    const uint32_t idx_offset_0 = idx;
    const uint32_t idx_offset_1 = idx_offset_0 + img_stride;
    const uint32_t idx_offset_2 = idx_offset_1 + img_stride;
    const uint32_t idx_offset_3 = idx_offset_2 + img_stride;

    const float fg_r = rgba_fg[idx_offset_0];
    const float fg_g = rgba_fg[idx_offset_1];
    const float fg_b = rgba_fg[idx_offset_2];
    const float fg_a = rgba_fg[idx_offset_3];

    const float bg_r = rgba_bg[idx_offset_0];
    const float bg_g = rgba_bg[idx_offset_1];
    const float bg_b = rgba_bg[idx_offset_2];
    const float bg_a = rgba_bg[idx_offset_3];

    const float out_a = fg_a + bg_a * (1.0f - fg_a);
    rgba_out[idx_offset_3] = out_a;

    if (out_a > 0.0f) {
        const float out_r = (fg_r * fg_a + bg_r * bg_a * (1.0f - fg_a)) / out_a;
        const float out_g = (fg_g * fg_a + bg_g * bg_a * (1.0f - fg_a)) / out_a;
        const float out_b = (fg_b * fg_a + bg_b * bg_a * (1.0f - fg_a)) / out_a;

        rgba_out[idx_offset_0] = out_r;
        rgba_out[idx_offset_1] = out_g;
        rgba_out[idx_offset_2] = out_b;
    }
}

TURBO_NAMESPACE_END
