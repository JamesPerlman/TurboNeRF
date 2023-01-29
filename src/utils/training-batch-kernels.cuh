#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../models/camera.cuh"
#include "../utils/color-utils.cuh"
#include "../utils/linalg.cuh"

NRC_NAMESPACE_BEGIN

/** This file contains helper kernels for generating rays and samples to fill the batch with data.
  */

__global__ void stbi_uchar_to_float(
	const uint32_t n_elements,
	const stbi_uc* __restrict__ src,
	float* __restrict__ dst
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < n_elements) {
		dst[idx] = (float)src[idx] / 255.0f;
	}
}

__global__ void generate_training_image_indices(
	const uint32_t n_elements,
	const uint32_t n_images,
	uint32_t* __restrict__ image_indices
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= n_elements) return;
	
	image_indices[idx] = idx % n_images;
}

__global__ void resize_floats_to_uint32_with_max(
	const uint32_t n_elements,
	const float* __restrict__ floats,
	uint32_t* __restrict__ uints,
	const float range_max
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= n_elements) return;
	
	float resized_val = floats[idx] * range_max;
	uints[idx] = (uint32_t)resized_val;
}

// generates rays and RGBs for training, assigns them to an array of contiguous data
__global__ void initialize_training_rays_and_pixels_kernel(
	const uint32_t n_rays,
	const uint32_t batch_size,
	const uint32_t n_images,
	const uint32_t image_data_stride,
	const int2 image_dimensions,
	const BoundingBox* __restrict__ bbox,
	const Camera* __restrict__ cameras,
	const stbi_uc* __restrict__ image_data,
	const uint32_t* __restrict__ img_index,
	const uint32_t* __restrict__ pix_index,

	// output buffers
	float* __restrict__ pix_rgba,
	float* __restrict__ ori_xyz,
	float* __restrict__ dir_xyz,
	float* __restrict__ idir_xyz,
	float* __restrict__ ray_t,
	bool* __restrict__ ray_alive
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_rays) return;

	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;
	const uint32_t i_offset_3 = i_offset_2 + batch_size;
	
	const uint32_t image_idx = img_index[i];
	const uint32_t pixel_idx = pix_index[i];
	
	const uint32_t pixel_x = pixel_idx % image_dimensions.x;
	const uint32_t pixel_y = pixel_idx / image_dimensions.x;
	const uint32_t x = pixel_x;
	const uint32_t y = pixel_y;
	const Camera cam = cameras[image_idx];
	
	const uint32_t img_offset = image_idx * image_data_stride;

	const stbi_uc* __restrict__ pixel = image_data + img_offset + 4 * pixel_idx;
	const stbi_uc r = pixel[0];
	const stbi_uc g = pixel[1];
	const stbi_uc b = pixel[2];
	const stbi_uc a = pixel[3];
	
	pix_rgba[i_offset_0] = srgb_to_linear((float)r / 255.0f);
	pix_rgba[i_offset_1] = srgb_to_linear((float)g / 255.0f);
	pix_rgba[i_offset_2] = srgb_to_linear((float)b / 255.0f);
	pix_rgba[i_offset_3] = srgb_to_linear((float)a / 255.0f);
	
	// TODO: optimize
	Ray local_ray = cam.local_ray_at_pixel_xy(x, y);

	float3 global_origin = cam.transform * local_ray.o;
	float3 global_direction = cam.transform.mmul_ul3x3(local_ray.d);
	
	ori_xyz[i_offset_0] = global_origin.x;
	ori_xyz[i_offset_1] = global_origin.y;
	ori_xyz[i_offset_2] = global_origin.z;

	// normalize ray directions
	const float n = rnorm3df(global_direction.x, global_direction.y, global_direction.z);

	const float dir_x = n * global_direction.x;
	const float dir_y = n * global_direction.y;
	const float dir_z = n * global_direction.z;
	
	const float idir_x = 1.0f / dir_x;
	const float idir_y = 1.0f / dir_y;
	const float idir_z = 1.0f / dir_z;
	
	dir_xyz[i_offset_0] = dir_x;
	dir_xyz[i_offset_1] = dir_y;
	dir_xyz[i_offset_2] = dir_z;

	idir_xyz[i_offset_0] = idir_x;
	idir_xyz[i_offset_1] = idir_y;
	idir_xyz[i_offset_2] = idir_z;
	
	float t;
	const bool intersects_bbox = bbox->get_ray_t_intersection(
		global_origin.x, global_origin.y, global_origin.z,
		dir_x, dir_y, dir_z,
		idir_x, idir_y, idir_z,
		t
	);

	ray_alive[i] = intersects_bbox;
	ray_t[i] = intersects_bbox ? fmaxf(0.0f, t + 1e-5f) : 0.0f;
}

// CONSIDER: move rays inside bounding box first?

__global__ void march_and_count_steps_per_ray_kernel(
	uint32_t n_rays,
	uint32_t batch_size,
	const BoundingBox* bbox,
	const CascadedOccupancyGrid* occ_grid,
	const float cone_angle,
	const float dt_min,
	const float dt_max,
	// input buffers
	const float* __restrict__ dir_xyz,
	const float* __restrict__ idir_xyz,
	// output/mixed use buffers
	bool* __restrict__ ray_alive,
	float* __restrict__ ori_xyz,
	float* __restrict__ ray_t,
	uint32_t* __restrict__ n_steps // one per ray
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= n_rays) return;

	if (!ray_alive[i]) {
		n_steps[i] = 0;
		return;
	};

	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;
	
	const float o_x = ori_xyz[i_offset_0];
	const float o_y = ori_xyz[i_offset_1];
	const float o_z = ori_xyz[i_offset_2];
	
	const float d_x = dir_xyz[i_offset_0];
	const float d_y = dir_xyz[i_offset_1];
	const float d_z = dir_xyz[i_offset_2];
	
	const float id_x = idir_xyz[i_offset_0];
	const float id_y = idir_xyz[i_offset_1];
	const float id_z = idir_xyz[i_offset_2];

	uint32_t n_steps_taken = 0;
	
	float t = ray_t[i];

	while (true) {
		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bbox->contains(x, y, z)) {
			break;
		}

		const float dt = occ_grid->get_dt(t, cone_angle, dt_min, dt_max);
		const int grid_level = occ_grid->get_grid_level_at(x, y, z, dt);

		if (occ_grid->is_occupied_at(grid_level, x, y, z)) {

			t += dt;

			if (n_steps_taken == 0) {
				// on first hit of an occupied cell, move ray origin to this cell
				ray_t[i] = 0.0f;
				ori_xyz[i_offset_0] = x;
				ori_xyz[i_offset_1] = y;
				ori_xyz[i_offset_2] = z;
			}

			++n_steps_taken;
		} else {
			// otherwise we need to find the next occupied cell
			// TODO: feed in normalized positions so we don't have to calculate them here!
			t += occ_grid->get_dt_to_next_voxel(
				x, y, z,
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				dt_min,
				grid_level
			);
		}
	}

	if (n_steps_taken == 0) {
		ray_alive[i] = false;
	}

	n_steps[i] = n_steps_taken;
}

/**
 * This kernel has a few purposes:
 * 1. March rays through the occupancy grid and generate start/end intervals for each sample
 * 2. Compact other training buffers to maximize coalesced memory accesses
 */

__global__ void march_and_generate_samples_and_compact_buffers_kernel(
	uint32_t batch_size,
	const BoundingBox* bbox,
	const float inv_aabb_size,
	const CascadedOccupancyGrid* occ_grid,
	const float dt_min,
	const float dt_max,
	const float cone_angle,
	
	// input buffers
	const float* __restrict__ in_ori_xyz,
	const float* __restrict__ in_dir_xyz,
	const float* __restrict__ in_idir_xyz,
	const float* __restrict__ in_ray_t,
	const uint32_t* __restrict__ n_ray_steps, // one per ray
	const uint32_t* __restrict__ n_steps_cum, // one per ray
	const bool* __restrict__ ray_alive,

	// output buffers
	float* __restrict__ out_pos_xyz,
	float* __restrict__ out_dir_xyz,
	float* __restrict__ out_dt
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= batch_size) return;

	if (!ray_alive[i]) return;

	// if the total number of cumulative steps is greater than the number of rays, we exit early to avoid writing outside of our sample buffers
	const uint32_t n_total_steps_cum = n_steps_cum[i];
	const uint32_t n_steps = n_ray_steps[i];

	if (n_total_steps_cum >= batch_size) return;

	// References to input buffers

	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;

	const float o_x = in_ori_xyz[i_offset_0];
	const float o_y = in_ori_xyz[i_offset_1];
	const float o_z = in_ori_xyz[i_offset_2];

	const float d_x = in_dir_xyz[i_offset_0];
	const float d_y = in_dir_xyz[i_offset_1];
	const float d_z = in_dir_xyz[i_offset_2];
	
	const float id_x = in_idir_xyz[i_offset_0];
	const float id_y = in_idir_xyz[i_offset_1];
	const float id_z = in_idir_xyz[i_offset_2];

	/** n_total_steps_cum is the cumulative number of steps taken by any ray up to and including ray i
	  * to get the offset of the data buffer holding samples for this ray,
	  * we must subtract the number of steps taken by this ray.
	  */
	
	const uint32_t sample_offset = n_total_steps_cum - n_steps;

	// Perform raymarching

	float t = in_ray_t[i];
	uint32_t n_steps_taken = 0;

	while (n_steps_taken < n_steps) {

		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bbox->contains(x, y, z)) {
			break;
		}

		const float dt = occ_grid->get_dt(t, cone_angle, dt_min, dt_max);
		const int grid_level = occ_grid->get_grid_level_at(x, y, z, dt);

		if (occ_grid->is_occupied_at(grid_level, x, y, z)) {
			t += dt;

			/**
			 * Here is where we assign training data to our compacted sample buffers.
			 * RIP coalesced memory accesses :(
			 * Worth it tho, gg ez.
			 */

			const uint32_t step_offset_0 = sample_offset + n_steps_taken;
			const uint32_t step_offset_1 = step_offset_0 + batch_size;
			const uint32_t step_offset_2 = step_offset_1 + batch_size;

			// generate network inputs
			out_dt[step_offset_0] = dt * inv_aabb_size;

			out_pos_xyz[step_offset_0] = x * inv_aabb_size + 0.5f;
			out_pos_xyz[step_offset_1] = y * inv_aabb_size + 0.5f;
			out_pos_xyz[step_offset_2] = z * inv_aabb_size + 0.5f;

			out_dir_xyz[step_offset_0] = (d_x + 1.0f) * 0.5f;
			out_dir_xyz[step_offset_1] = (d_y + 1.0f) * 0.5f;
			out_dir_xyz[step_offset_2] = (d_z + 1.0f) * 0.5f;

			++n_steps_taken;
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
}

NRC_NAMESPACE_END
