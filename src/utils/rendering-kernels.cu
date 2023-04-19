
#include <device_launch_parameters.h>
#include <stbi/stb_image.h>

#include "rendering-kernels.cuh"
#include "../core/occupancy-grid.cuh"
#include "../math/geometric-intersections.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../utils/color-utils.cuh"
#include "../utils/common-network-kernels.cuh"
#include "../utils/nerf-constants.cuh"

using namespace tcnn;

TURBO_NAMESPACE_BEGIN

__global__ void prepare_for_linear_raymarching_kernel(
    const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t n_nerfs,
	const OccupancyGrid* __restrict__ grids,
	const BoundingBox* __restrict__ bboxes,
	const Transform4f* __restrict__ transforms,
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

    // make sure this ray intersects some NeRF's bbox
	// tmin = float max
	float t_min = FLT_MAX;
	float t_max = 0.0f;
	int nearest_nerf = -1;

	uint32_t nerf_offset = 0;
	for (int n = 0; n < n_nerfs; ++n) {

		const uint32_t nerf_ray_idx = nerf_offset + i;

		const BoundingBox& bbox = bboxes[n];
		const Transform4f& transform = transforms[n];
		const Transform4f& itrans = transform.inverse();

		// we need to transform the ray into the NeRF's local space
		float3 o = itrans * float3{ o_x, o_y, o_z };
		float3 d = normalized(itrans.mmul_ul3x3(float3{ d_x, d_y, d_z }));
		float3 id = float3{ 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };

		float _tmin;
		float _tmax;

		const bool intersects_bbox = bbox.get_ray_t_intersections(
			o.x, o.y, o.z,
			d.x, d.y, d.z,
			id.x, id.y, id.z,
			_tmin, _tmax
		);

		if (!intersects_bbox) {
			continue;
		}

		_tmin = fmaxf(_tmin, 1e-4f);

		const uint32_t ixn_idx = (n / 32) * batch_size + i;
		intersectors[ixn_idx] |= (1 << (n % 32));

		if (_tmin < t_min) {
			t_min = _tmin;
			nearest_nerf = n;
		}

		if (_tmax > t_max) {
			t_max = _tmax;
		}

		// set min and max for this NeRF's ray
		nerf_ray_t[nerf_ray_idx] = _tmin;
		nerf_tmax[nerf_ray_idx] = _tmax;

		// there can only be one active ray
		nerf_ray_active[nerf_ray_idx] = false;
		
		nerf_offset += batch_size;
	}

	if (nearest_nerf == -1) {
		// ray intersects zero NeRFs, and thus can be terminated immediately
		ray_alive[i] = false;
		return;
	}

	uint32_t nearest_nerf_idx = (uint32_t)nearest_nerf * batch_size + i;
	nerf_ray_active[nearest_nerf_idx] = true;

	// t_max is the greatest value less than or equal to ray_tmax[i]

	ray_tmax[i] = fminf(t_max, ray_tmax[i]);
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
	float* __restrict__ ray_tmax,
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
		
		if (!cam.show_image_planes) {
			continue;
		}

		const Transform4f c2w = cam.transform;
		const Transform4f w2c = c2w.inverse();
		const float3 c2w_xyz = c2w.get_translation();
		const float3 r2{ c2w.m02, c2w.m12, c2w.m22 };
		const float m = l2_norm(r2);
		const float3 plane_normal = r2 / m;
		const float2 base_size{
			cam.resolution_f.x / cam.focal_length.x,
			cam.resolution_f.y / cam.focal_length.y
		};

		if (show_near_planes) {
			const float3 near_center = cam.near * plane_normal + c2w_xyz;
			const float2 near_size = (cam.near / m) * base_size;	

			float t_near;
			float2 uv_near;

			bool intersects_near = ray_plane_intersection(
				ray_o,
				ray_d,
				near_center,
				plane_normal,
				near_size,
				w2c,
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
			const float3 far_center = cam.far * plane_normal + c2w_xyz;
			const float2 far_size = (cam.far / m) * base_size;	

			float t_far;
			float2 uv_far;

			bool intersects_far = ray_plane_intersection(
				ray_o,
				ray_d,
				far_center,
				plane_normal,
				far_size,
				w2c,
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

		const int pix_ix = (int)(t_min_uv.x * (float)training_img_dims.x);
		const int pix_iy = (int)(t_min_uv.y * (float)training_img_dims.y);

		// clamp to the image bounds
		const size_t pix_x = (size_t)clamp(pix_ix, 0, training_img_dims.x - 1);
		const size_t pix_y = (size_t)clamp(pix_iy, 0, training_img_dims.y - 1);

		// get the pixel index
		const size_t train_pix_offset = n_pix_per_training_img * (size_t)t_min_cam_idx;
		const size_t pix_w = (size_t)training_img_dims.x;
		const size_t train_pix_idx = pix_y * pix_w + pix_x;

		const stbi_uc* train_rgba = train_img_data + 4 * (train_pix_offset + train_pix_idx);
		
		// write the pixel
		out_rgba_buf[out_idx_offset_0] = __srgb_to_linear((float)train_rgba[0] / 255.0f);
		out_rgba_buf[out_idx_offset_1] = __srgb_to_linear((float)train_rgba[1] / 255.0f);
		out_rgba_buf[out_idx_offset_2] = __srgb_to_linear((float)train_rgba[2] / 255.0f);
		out_rgba_buf[out_idx_offset_3] = (float)train_rgba[3] / 255.0f;

		// set t_max
		ray_tmax[idx] = t_min;
	} else {
		// clear output pixel
		out_rgba_buf[out_idx_offset_0] = 0.0f;
		out_rgba_buf[out_idx_offset_1] = 0.0f;
		out_rgba_buf[out_idx_offset_2] = 0.0f;
		out_rgba_buf[out_idx_offset_3] = 0.0f;
	}
}

/**
 * The multi-nerf raymarching algorithm goes as follows:
 * 1. Filter only nerfs that intersect the ray
 * 2. For each active ray, march it forward until it hits a voxel
 * 3. Assign a new t-value to the nerf's ray
 * 4. The nerf ray with the smallest t-value becomes active
 * 5. Generate a sample point for that ray so it will be rendered.
 * 
 */

__global__ void march_rays_and_generate_network_inputs_kernel(
    const uint32_t n_rays,
	const uint32_t n_nerfs,
	const uint32_t batch_size,
	const uint32_t network_batch,
	const int n_steps_max,
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
) {
	// get thread index
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (idx >= n_rays) return;

    // check if ray has terminated
    if (!ray_alive[idx]) return;

	// References to input buffers
	const uint32_t idx_offset_0 = idx;
	const uint32_t idx_offset_1 = idx_offset_0 + batch_size;
	const uint32_t idx_offset_2 = idx_offset_1 + batch_size;

	const float o_x = ray_ori[idx_offset_0];
	const float o_y = ray_ori[idx_offset_1];
	const float o_z = ray_ori[idx_offset_2];

	const float d_x = ray_dir[idx_offset_0];
	const float d_y = ray_dir[idx_offset_1];
	const float d_z = ray_dir[idx_offset_2];
	
	// Perform raymarching
	float global_tmax = ray_tmax[idx];

	int n_steps = 0;

	while (n_steps < n_steps_max) {

		// Local variables for output buffers
		float nearest_pos_x, nearest_pos_y, nearest_pos_z;
		float nearest_dir_x, nearest_dir_y, nearest_dir_z;
		float nearest_dt;

		float nearest_inv_aabb_size = 0.0f;
		float nearest_t = FLT_MAX;
		int nearest_nerf = -1;

		// first we need to march each nerf's ray to the next occupied voxel
		for (int n = 0; n < n_nerfs; ++n) {
			const uint32_t nerf_ray_idx = n * batch_size + idx;
			
			// check if this ray even intersects this NeRF
			const uint32_t ixn_idx = (n / 32) * batch_size + idx;
			const bool intersects = intersectors[ixn_idx] & (1 << (n % 32));
			if (!intersects) {
				continue;
			}

			// we need to find the first NeRF that this ray intersects

			float t = nerf_ray_t[nerf_ray_idx];
			const float t_max = nerf_tmax[nerf_ray_idx];

			// is the ray position currently inside this NeRF?
			if (t > t_max) {
				continue;
			}

			// TODO: shared memory

			const OccupancyGrid& grid = grids[n];
			const BoundingBox& bbox = bboxes[n];
			const Transform4f itrans = transforms[n].inverse();

			const float dt_max = dt_min * bbox.size();
			const float inv_aabb_size = 1.0f / bbox.size();

			const float3 o = itrans * float3{o_x, o_y, o_z};
			const float3 d = normalized(itrans.mmul_ul3x3(float3{d_x, d_y, d_z}));
			const float3 id{ 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };

			/**
			 * this nerf will be active if its grid is occupied at the current t value
			 * we can only march each ray by one step using this technique
			 * but it does get the job done!
			 */

			const float abs_t_max = fminf(t_max, global_tmax);

			float x, y, z;
			float dt;

			// only march if ray is active
			if (nerf_ray_active[nerf_ray_idx]) {

				do {
					x = o.x + t * d.x;
					y = o.y + t * d.y;
					z = o.z + t * d.z;

					dt = grid.get_dt(t, cone_angle, dt_min, dt_max);
					const int grid_level = grid.get_grid_level_at(x, y, z, dt);

					if (grid.is_occupied_at(grid_level, x, y, z)) {

						t += dt;

						break;
					} else {
						// otherwise we need to find the next occupied cell
						t += grid.get_dt_to_next_voxel(
							x, y, z,
							d.x, d.y, d.z,
							id.x, id.y, id.z,
							dt_min,
							grid_level
						);
					}

				} while (t < abs_t_max);
				
				// update nerf t value
				nerf_ray_t[nerf_ray_idx] = t;
				
				// only one nerf may be active at a time
				nerf_ray_active[nerf_ray_idx] = false;
			
			} // if (nerf_ray_active[nerf_ray_idx])
			else {
				x = o.x + t * d.x;
				y = o.y + t * d.y;
				z = o.z + t * d.z;

				dt = grid.get_dt(t, cone_angle, dt_min, dt_max);
			}

			if (t < nearest_t) {
				nearest_t = t;
				nearest_nerf = n;

				nearest_inv_aabb_size = inv_aabb_size;
				nearest_pos_x = x; nearest_pos_y = y; nearest_pos_z = z;
				nearest_dir_x = d.x; nearest_dir_y = d.y; nearest_dir_z = d.z;
				nearest_dt = dt;
			}
		} // for (int n = 0; n < n_nerfs; ++n)

		if (nearest_nerf > -1) {

			const uint32_t sample_offset_0 = n_rays * n_steps + idx;
			const uint32_t sample_offset_1 = sample_offset_0 + network_batch;
			const uint32_t sample_offset_2 = sample_offset_1 + network_batch;

			network_pos[sample_offset_0] = nearest_inv_aabb_size * nearest_pos_x + 0.5f;
			network_pos[sample_offset_1] = nearest_inv_aabb_size * nearest_pos_y + 0.5f;
			network_pos[sample_offset_2] = nearest_inv_aabb_size * nearest_pos_z + 0.5f;

			network_dir[sample_offset_0] = 0.5f * nearest_dir_x + 0.5f;
			network_dir[sample_offset_1] = 0.5f * nearest_dir_y + 0.5f;
			network_dir[sample_offset_2] = 0.5f * nearest_dir_z + 0.5f;

			network_dt[sample_offset_0] = nearest_inv_aabb_size * nearest_dt;

			sample_nerf_id[sample_offset_0] = nearest_nerf;

			nerf_ray_active[nearest_nerf * batch_size + idx] = true;
		} else {
			// no nerf is active, the ray must die.
			ray_alive[idx] = false;
			break;
		}

		++n_steps;
	}

	// we must set the remaining sample_nerf_id values to -1
	for (int i = n_steps; i < n_steps_max; ++i) {
		const uint32_t sample_offset = n_rays * i + idx;
		sample_nerf_id[sample_offset] = -1;
	}

	n_steps_total[idx] = n_steps;
}

// sample compaction
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
) {
	const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_idx >= n_compacted_samples) return;

	// 3-component buffers
	int c_offset = c_idx;
	int e_offset = indices[c_idx];

	#pragma unroll
	for (int i = 0; i < 3; ++i) {
		out_network_pos[c_offset] = in_network_pos[e_offset];
		out_network_dir[c_offset] = in_network_dir[e_offset];

		c_offset += new_batch_size;
		e_offset += old_batch_size;
	}
}

// sample re-expansion
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
) {
	const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (c_idx >= n_compacted_samples) return;

	int c_offset = c_idx;
	int e_offset = indices[c_idx];

	out_network_density[e_offset] = in_network_density[c_offset];

	#pragma unroll
	for (int i = 0; i < 3; ++i) {
		out_network_rgb[e_offset] = in_network_rgb[c_offset];

		c_offset += new_batch_size;
		e_offset += old_batch_size;
	}
}

// alpha compositing kernel, composites the latest samples into the output image
__global__ void composite_samples_kernel(
	const uint32_t n_rays,
	const uint32_t network_stride,
	const uint32_t output_stride,
	const uint32_t n_steps_max,

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
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

	if (ray_alive[idx] == false) return;

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

	const uint32_t n_steps = n_steps_total[idx];
	float trans = ray_trans[idx];
	
	for (uint32_t step = 0; step < n_steps; ++step) {
			
		const uint32_t net_idx_0 = n_rays * step + idx;
		const uint32_t net_idx_1 = net_idx_0 + network_stride;
		const uint32_t net_idx_2 = net_idx_1 + network_stride;

		const float net_r = (float)network_rgb[net_idx_0];
		const float net_g = (float)network_rgb[net_idx_1];
		const float net_b = (float)network_rgb[net_idx_2];

		const float alpha = density_to_alpha(
			network_density[net_idx_0],
			ray_dt[net_idx_0]
		);

		const float weight = alpha * trans;

		// composite the same way we do accumulation during training
		out_r += weight * net_r;
		out_g += weight * net_g;
		out_b += weight * net_b;
		out_a += weight;

		// update and threshold transmittance
		trans *= 1.0f - alpha;

		output_rgba[idx_offset_0] = out_r;
		output_rgba[idx_offset_1] = out_g;
		output_rgba[idx_offset_2] = out_b;
		output_rgba[idx_offset_3] = out_a;

		if (trans < NeRFConstants::min_transmittance) {
			ray_alive[idx] = false;
			return;
		}
	}

	ray_trans[idx] = trans;
}

// ray compaction
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
) {
    // compacted index is the index to write to
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= n_compacted_rays) return;

	// expanded index is the index to read from
	const int e_idx = indices[c_idx];

	// 1-component buffers (global)
	out_idx[c_idx]		= in_idx[e_idx];
	out_ray_tmax[c_idx]	= in_ray_tmax[e_idx];
	out_trans[c_idx]	= in_trans[e_idx];

	int c_offset = c_idx;
	int e_offset = e_idx;

	#pragma unroll
	for (int i = 0; i < 3; ++i) {
		out_ori[c_offset] = in_ori[e_offset];
		out_dir[c_offset] = in_dir[e_offset];

		c_offset += batch_size;
		e_offset += batch_size;
	}

	// nerf buffers
	int c_nerf_offset = c_idx;
	int e_nerf_offset = e_idx;

	for (int n = 0; n < n_nerfs; ++n) {
		out_nerf_ray_active[c_nerf_offset]	= in_nerf_ray_active[e_nerf_offset];
		out_nerf_ray_t[c_nerf_offset]		= in_nerf_ray_t[e_nerf_offset];
		out_nerf_ray_tmax[c_nerf_offset]	= in_nerf_ray_tmax[e_nerf_offset];
		
		c_nerf_offset += batch_size;
		e_nerf_offset += batch_size;
	}

	// intersector buffers
	int c_intx_offset = c_idx;
	int e_intx_offset = e_idx;

	for (int i = 0; i <= n_nerfs / 32; ++i) {
		out_intersectors[c_intx_offset] = in_intersectors[e_intx_offset];

		c_intx_offset += batch_size;
		e_intx_offset += batch_size;
	}
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
