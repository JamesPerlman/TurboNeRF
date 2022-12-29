#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Dense>
#include <stbi/stb_image.h>

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
	
	image_indices[idx] = idx * n_images / n_elements;
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
	const uint32_t n_batch_elements,
	const uint32_t n_images,
	const uint32_t image_data_stride,
	const Eigen::Vector2i image_dimensions,
	const Camera* __restrict__ cameras,
	const stbi_uc* __restrict__ image_data,
	const uint32_t* __restrict__ img_index,
	const uint32_t* __restrict__ pix_index,
	float* __restrict__ pix_r, float* __restrict__ pix_g, float* __restrict__ pix_b, float* __restrict__ pix_a,
	float* __restrict__ ori_x, float* __restrict__ ori_y, float* __restrict__ ori_z,
	float* __restrict__ dir_x, float* __restrict__ dir_y, float* __restrict__ dir_z,
	float* __restrict__ idir_x, float* __restrict__ idir_y, float* __restrict__ idir_z
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_batch_elements) return;
	
	uint32_t image_idx = img_index[idx];
	uint32_t pixel_idx = pix_index[idx];
	
	uint32_t pixel_x = pixel_idx % image_dimensions.x();
	uint32_t pixel_y = pixel_idx / image_dimensions.x();
	uint32_t x = pixel_x;
	uint32_t y = pixel_y;
	Camera cam = cameras[image_idx];
	
	uint32_t img_offset = image_idx * image_data_stride;

	const stbi_uc* __restrict__ pixel = image_data + img_offset + 3 * pixel_idx;
	stbi_uc r = pixel[0];
	stbi_uc g = pixel[1];
	stbi_uc b = pixel[2];
	stbi_uc a = pixel[3];
	
	pix_r[idx] = (float)r / 255.0f;
	pix_g[idx] = (float)g / 255.0f;
	pix_b[idx] = (float)b / 255.0f;
	pix_a[idx] = (float)a / 255.0f;
	
	Ray ray = cam.get_ray_at_pixel_xy(x, y);

	ori_x[idx] = ray.o.x();
	ori_y[idx] = ray.o.y();
	ori_z[idx] = ray.o.z();

	dir_x[idx] = ray.d.x();
	dir_y[idx] = ray.d.y();
	dir_z[idx] = ray.d.z();

	idir_x[idx] = 1.0f / ray.d.x();
	idir_y[idx] = 1.0f / ray.d.y();
	idir_z[idx] = 1.0f / ray.d.z();
}

// CONSIDER: move rays inside bounding box first?

__global__ void march_and_count_steps_per_ray_kernel(
	uint32_t n_rays,
	const BoundingBox* bounding_box,
	const CascadedOccupancyGrid* occupancy_grid,
	const float cone_angle,
	const float dt_min,
	const float dt_max,
	const float* __restrict__ ori_x, const float* __restrict__ ori_y, const float* __restrict__ ori_z,
	const float* __restrict__ dir_x, const float* __restrict__ dir_y, const float* __restrict__ dir_z,
	const float* __restrict__ idir_x, const float* __restrict__ idir_y, const float* __restrict__ idir_z,
	uint32_t* __restrict__ n_steps // one per ray
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= n_rays) return;
	
	const float o_x = ori_x[i];
	const float o_y = ori_y[i];
	const float o_z = ori_z[i];

	const float d_x = dir_x[i];
	const float d_y = dir_y[i];
	const float d_z = dir_z[i];
	
	const float id_x = idir_x[i];
	const float id_y = idir_y[i];
	const float id_z = idir_z[i];

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
	uint32_t n_rays,
	uint32_t n_steps_max,
	const BoundingBox* bounding_box,
	const CascadedOccupancyGrid* occupancy_grid,
	const float dt_min,
	const float dt_max,
	const float cone_angle,
	
	// input buffers
	const float* __restrict__ in_pix_r, const float* __restrict__ in_pix_g, const float* __restrict__ in_pix_b, const float* __restrict__ in_pix_a,
	const float* __restrict__ in_ori_x, const float* __restrict__ in_ori_y, const float* __restrict__ in_ori_z,
	const float* __restrict__ in_dir_x, const float* __restrict__ in_dir_y, const float* __restrict__ in_dir_z,
	const float* __restrict__ in_idir_x, const float* __restrict__ in_idir_y, const float* __restrict__ in_idir_z,
	const uint32_t* __restrict__ n_ray_steps, 		// one per ray
	const uint32_t* __restrict__ n_steps_cum, // one per ray

	// output buffers
	float* __restrict__ out_pix_r, float* __restrict__ out_pix_g, float* __restrict__ out_pix_b, float* __restrict__ out_pix_a,
	float* __restrict__ out_ori_x, float* __restrict__ out_ori_y, float* __restrict__ out_ori_z,
	float* __restrict__ out_dir_x, float* __restrict__ out_dir_y, float* __restrict__ out_dir_z,
	float* __restrict__ out_t0, float* __restrict__ out_t1
) {
	// get thread index
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// check if thread is out of bounds
	if (i >= n_rays) return;


	// if the total number of cumulative steps is greater than the number of rays, we exit early to avoid overflowing any buffers
	const uint32_t n_total_steps_cum = n_steps_cum[i];

	if (n_total_steps_cum >= n_rays) return;

	// References to input buffers

	const float o_x = in_ori_x[i];
	const float o_y = in_ori_y[i];
	const float o_z = in_ori_z[i];

	const float d_x = in_dir_x[i];
	const float d_y = in_dir_y[i];
	const float d_z = in_dir_z[i];
	
	const float id_x = in_idir_x[i];
	const float id_y = in_idir_y[i];
	const float id_z = in_idir_z[i];

	/** n_total_steps_cum is the cumulative number of steps taken by any ray up to and including ray i
	  * to get the offset of the data buffer holding samples for this ray,
	  * we must subtract the number of steps taken by this ray.
	  */
	
	uint32_t sample_offset = n_total_steps_cum - n_ray_steps[i];

	/**
	 * Store pointers to the sub-location for compacted data to be placed
	 * It's actually kinda beautiful :')
	 */
	float* __restrict__ compacted_pix_r = out_pix_r + sample_offset;
	float* __restrict__ compacted_pix_g = out_pix_g + sample_offset;
	float* __restrict__ compacted_pix_b = out_pix_b + sample_offset;
	float* __restrict__ compacted_pix_a = out_pix_a + sample_offset;
	
	float* __restrict__ compacted_ori_x = out_ori_x + sample_offset;
	float* __restrict__ compacted_ori_y = out_ori_y + sample_offset;
	float* __restrict__ compacted_ori_z = out_ori_z + sample_offset;
	
	float* __restrict__ compacted_dir_x = out_dir_x + sample_offset;
	float* __restrict__ compacted_dir_y = out_dir_y + sample_offset;
	float* __restrict__ compacted_dir_z = out_dir_z + sample_offset;

	float* __restrict__ compacted_t0 = out_t0 + sample_offset;
	float* __restrict__ compacted_t1 = out_t1 + sample_offset;

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

			// assign start/end t-values for this sampling interval
			// t0 (t_start) is our most recent t-value
			compacted_t0[n_steps_taken] = t;

			// march t forward
			t += dt;

			// t1 (t_end) is our new t-value
			compacted_t1[n_steps_taken] = t;

			/**
			 * Compact the rest of the buffers.
			 * We use the minimum number of buffers required because we prefer using coalesced memory access.
			 * We will use another kernel to transform this data further before passing it to the neural network.
			 * After this step we will still need to stratify the t-values and generate the sample positions.
			 */

			compacted_pix_r[sample_offset] = in_pix_r[i];
			compacted_pix_g[sample_offset] = in_pix_g[i];
			compacted_pix_b[sample_offset] = in_pix_b[i];
			compacted_pix_a[sample_offset] = in_pix_a[i];

			compacted_ori_x[sample_offset] = o_x;
			compacted_ori_y[sample_offset] = o_y;
			compacted_ori_z[sample_offset] = o_z;

			compacted_dir_x[sample_offset] = d_x;
			compacted_dir_y[sample_offset] = d_y;
			compacted_dir_z[sample_offset] = d_z;

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
	uint32_t n_samples,
	const float* __restrict__ t0, const float* __restrict__ t1,
	const float* __restrict__ random_floats,
	float* __restrict__ out_x, float* __restrict__ out_y, float* __restrict__ out_z
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_samples) {
		return;
	}

	const float t0_i = t0[i];
	const float t1_i = t1[i];

	const float random_float = random_floats[i];

	const float t = t0_i + (t1_i - t0_i) * random_float;

	out_x[i] = t;
	out_y[i] = t;
	out_z[i] = t;
}

NRC_NAMESPACE_END
