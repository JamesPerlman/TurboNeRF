#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>
#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/bounding-box.cuh"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../models/camera.h"

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
	const uint32_t batch_size,
	const uint32_t n_images,
	const uint32_t image_data_stride,
	const Eigen::Vector2i image_dimensions,
	const Camera* __restrict__ cameras,
	const stbi_uc* __restrict__ image_data,
	const uint32_t* __restrict__ img_index,
	const uint32_t* __restrict__ pix_index,
	float* __restrict__ pix_rgba,
	float* __restrict__ ori_xyz,
	float* __restrict__ dir_xyz,
	float* __restrict__ idir_xyz
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batch_size) return;

	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;
	const uint32_t i_offset_3 = i_offset_2 + batch_size;
	
	const uint32_t image_idx = img_index[i];
	const uint32_t pixel_idx = pix_index[i];
	
	const uint32_t pixel_x = pixel_idx % image_dimensions.x();
	const uint32_t pixel_y = pixel_idx / image_dimensions.x();
	const uint32_t x = pixel_x;
	const uint32_t y = pixel_y;
	const Camera cam = cameras[image_idx];
	
	const uint32_t img_offset = image_idx * image_data_stride;

	const stbi_uc* __restrict__ pixel = image_data + img_offset + 4 * pixel_idx;
	const stbi_uc r = pixel[0];
	const stbi_uc g = pixel[1];
	const stbi_uc b = pixel[2];
	const stbi_uc a = pixel[3];
	
	pix_rgba[i_offset_0] = (float)r / 255.0f;
	pix_rgba[i_offset_1] = (float)g / 255.0f;
	pix_rgba[i_offset_2] = (float)b / 255.0f;
	pix_rgba[i_offset_3] = (float)a / 255.0f;
	
	// TODO: optimize
	Ray ray = cam.get_ray_at_pixel_xy(x, y);

	ray.o = (cam.transform * ray.o.homogeneous()).head<3>();
	ray.d = (cam.transform * ray.d.homogeneous()).head<3>() - ray.o;

	ori_xyz[i_offset_0] = ray.o.x();
	ori_xyz[i_offset_1] = ray.o.y();
	ori_xyz[i_offset_2] = ray.o.z();

	// normalize ray directions
	const float n = rsqrtf(ray.d.x() * ray.d.x() + ray.d.y() * ray.d.y() + ray.d.z() * ray.d.z());

	const float ray_dx = ray.d.x() * n;
	const float ray_dy = ray.d.y() * n;
	const float ray_dz = ray.d.z() * n;

	dir_xyz[i_offset_0] = ray_dx;
	dir_xyz[i_offset_1] = ray_dy;
	dir_xyz[i_offset_2] = ray_dz;

	idir_xyz[i_offset_0] = 1.0f / ray_dx;
	idir_xyz[i_offset_1] = 1.0f / ray_dy;
	idir_xyz[i_offset_2] = 1.0f / ray_dz;
}

// CONSIDER: move rays inside bounding box first?

__global__ void march_and_count_steps_per_ray_kernel(
	uint32_t batch_size,
	const BoundingBox* bounding_box,
	const CascadedOccupancyGrid* occupancy_grid,
	const float cone_angle,
	const float dt_min,
	const float dt_max,
	const float* __restrict__ ori_xyz,
	const float* __restrict__ dir_xyz,
	const float* __restrict__ idir_xyz,
	uint32_t* __restrict__ n_steps // one per ray
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= batch_size) return;

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
	
	float t = 0.0f;

	while (true) {

		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bounding_box->contains(x, y, z)) {
			break;
		}

		int grid_level = occupancy_grid->get_grid_level_at(x, y, z, dt_min);

		if (occupancy_grid->is_occupied_at(grid_level, x, y, z)) {
			// if grid is occupied here, march forward by a calculated dt
			float dt = occupancy_grid->get_dt(t, cone_angle, dt_min, dt_max);
			t += dt;

			++n_steps_taken;
		} else {
			// otherwise we need to find the next occupied cell
			t = occupancy_grid->get_t_advanced_to_next_voxel(
				o_x, o_y, o_z,
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				t, dt_min
			);
		}
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
	const BoundingBox* bounding_box,
	const CascadedOccupancyGrid* occupancy_grid,
	const float dt_min,
	const float dt_max,
	const float cone_angle,
	
	// input buffers
	const float* __restrict__ in_pix_rgba,
	const float* __restrict__ in_ori_xyz,
	const float* __restrict__ in_dir_xyz,
	const float* __restrict__ in_idir_xyz,
	const uint32_t* __restrict__ n_ray_steps, // one per ray
	const uint32_t* __restrict__ n_steps_cum, // one per ray

	// output buffers
	float* __restrict__ out_ori_xyz,
	float* __restrict__ out_dir_xyz,
	float* __restrict__ out_t0,
	float* __restrict__ out_t1
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= batch_size) return;


	// if the total number of cumulative steps is greater than the number of rays, we exit early to avoid overflowing any buffers
	const uint32_t n_total_steps_cum = n_steps_cum[i];

	if (n_total_steps_cum >= batch_size) return;

	// References to input buffers
	const uint32_t batch_offset_0 = 0;
	const uint32_t batch_offset_1 = batch_size;
	const uint32_t batch_offset_2 = batch_size << 1; // this is presumably faster than (batch_size * 2) or (batch_offset_1 + batch_size)
	const uint32_t batch_offset_3 = batch_offset_2 + batch_size;

	const uint32_t i_offset_0 = i + batch_offset_0;
	const uint32_t i_offset_1 = i + batch_offset_1;
	const uint32_t i_offset_2 = i + batch_offset_2;
	const uint32_t i_offset_3 = i + batch_offset_3;

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
	
	const uint32_t sample_offset_0 = n_total_steps_cum - n_ray_steps[i];
	const uint32_t sample_offset_1 = sample_offset_0 + batch_offset_1;
	const uint32_t sample_offset_2 = sample_offset_0 + batch_offset_2;
	const uint32_t sample_offset_3 = sample_offset_0 + batch_offset_3;

	// Perform raymarching

	float t = 0.0f;
	uint32_t n_steps_taken = 0;

	while (true) {

		const float x = o_x + t * d_x;
		const float y = o_y + t * d_y;
		const float z = o_z + t * d_z;

		if (!bounding_box->contains(x, y, z)) {
			break;
		}

		int grid_level = occupancy_grid->get_grid_level_at(x, y, z, dt_min);

		if (occupancy_grid->is_occupied_at(grid_level, x, y, z)) {
			// if grid is occupied here, march forward by a calculated dt
			float dt = occupancy_grid->get_dt(t, cone_angle, dt_min, dt_max);

			/**
			 * Here is where we assign training data to our compacted sample buffers.
			 * RIP coalesced memory accesses :(
			 * Worth it tho, gg ez.
			 */

			const uint32_t step_offset_0 = sample_offset_0 + n_steps_taken;
			const uint32_t step_offset_1 = sample_offset_1 + n_steps_taken;
			const uint32_t step_offset_2 = sample_offset_2 + n_steps_taken;
			const uint32_t step_offset_3 = sample_offset_3 + n_steps_taken;

			// assign start/end t-values for this sampling interval
			// t0 (t_start) is our most recent t-value
			out_t0[step_offset_0] = t;

			// march t forward
			t += dt;

			// t1 (t_end) is our new t-value
			out_t1[step_offset_0] = t;

			/**
			 * Compact the rest of the buffers.
			 * We use the minimum number of buffers required because we prefer using coalesced memory access.
			 * We will use another kernel to transform this data further before passing it to the neural network.
			 * After this step we will still need to stratify the t-values and generate the sample positions.
			 */

			out_ori_xyz[step_offset_0] = o_x;
			out_ori_xyz[step_offset_1] = o_y;
			out_ori_xyz[step_offset_2] = o_z;

			out_dir_xyz[step_offset_0] = d_x;
			out_dir_xyz[step_offset_1] = d_y;
			out_dir_xyz[step_offset_2] = d_z;

			++n_steps_taken;

		} else {
			// otherwise we need to find the next occupied cell
			t = occupancy_grid->get_t_advanced_to_next_voxel(
				o_x, o_y, o_z,
				d_x, d_y, d_z,
				id_x, id_y, id_z,
				t, dt_min
			);
		}
	}
}

/**
 * This kernel uses the t0 and t1 values to generate the sample positions.
 * We stratify the sample points using a buffer of random offsets and interpolate between t0 and t1 linearly.
 */
__global__ void generate_stratified_sample_positions_kernel(
	uint32_t batch_size,
	const float* __restrict__ t0,
	const float* __restrict__ t1,
	const float* __restrict__ random_floats,
	const float* __restrict__ in_ori_xyz,
	const float* __restrict__ in_dir_xyz,
	float* __restrict__ out_xyz,
	float* __restrict__ out_dt
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= batch_size) {
		return;
	}

	// Grab local references to global data
	const uint32_t i_offset_0 = i;
	const uint32_t i_offset_1 = i_offset_0 + batch_size;
	const uint32_t i_offset_2 = i_offset_1 + batch_size;

	const float t0_i = t0[i_offset_0];
	const float t1_i = t1[i_offset_0];
	
	const float k = random_floats[i_offset_0];
	
	const float o_x = in_ori_xyz[i_offset_0];
	const float o_y = in_ori_xyz[i_offset_1];
	const float o_z = in_ori_xyz[i_offset_2];
	
	const float d_x = in_dir_xyz[i_offset_0];
	const float d_y = in_dir_xyz[i_offset_1];
	const float d_z = in_dir_xyz[i_offset_2];

	// Calculate sample position
	const float dt = t1_i - t0_i;
	const float t = t0_i + dt * k;

	out_xyz[i_offset_0] = o_x + t * d_x;
	out_xyz[i_offset_1] = o_y + t * d_y;
	out_xyz[i_offset_2] = o_z + t * d_z;

	out_dt[i_offset_0] = dt;
}

NRC_NAMESPACE_END
