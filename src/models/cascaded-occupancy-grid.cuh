/**
 * Most of this code is inspired by Müller, et al:
 * Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, 2022:
 * https://dl.acm.org/doi/pdf/10.1145/3528223.3530127
 *
 * Some of this code is inspired by NerfAcc (Copyright (c) 2022 Ruilong Li, UC Berkeley).
 * Please see LICENSES/NerfAcc.md for details (MIT License)
 */

#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

#include "../common.h"
#include "../utils/bit-utils.cuh"
#include "cascaded-occupancy-grid-workspace.cuh"

NRC_NAMESPACE_BEGIN

struct CascadedOccupancyGrid {
	const int n_levels;
	const float resolution_f;
	const float inv_resolution_f;
	const uint32_t resolution_i;
	const uint32_t volume_i;

	CascadedOccupancyGridWorkspace workspace;
	// level is the power of two domain size (K from pg. 15 of Müller, et al. 2022)
	// grid goes from [-2^(level - 1) + 0.5, 2^(level - 1) + 0.5] in each dimension
	CascadedOccupancyGrid(const int& n_levels, const int& resolution = 128)
		: n_levels(n_levels)
		, resolution_i(resolution)
		, resolution_f(resolution)
		, inv_resolution_f(1.0f / resolution_f)
		, volume_i(resolution * resolution * resolution)
	{};

	CascadedOccupancyGrid() = default;

	// class function to calculate number of bytes needed to store the grid
	static inline NRC_HOST_DEVICE size_t get_n_total_elements(
		const int& n_levels,
		const int& grid_resolution = 128
	) {
		const int side_length = next_power_of_two(grid_resolution);
		return (size_t)(n_levels * side_length * side_length * side_length);
	}
	
	// allocators/initializers
	uint8_t* initialize(const cudaStream_t& stream, const bool& use_full_precision_values) {
		workspace.enlarge(
			stream,
			CascadedOccupancyGrid::get_n_total_elements(n_levels, resolution_i),
			use_full_precision_values
		);
		return workspace.bitfield;
	}

	// pointer getters
	inline NRC_HOST_DEVICE uint8_t* get_bitfield() const {
		return workspace.bitfield;
	}

	inline NRC_HOST_DEVICE float* get_density() const {
		return workspace.values;
	}

	// memory setters
	inline __host__ void set_bitfield(const cudaStream_t& stream, const uint8_t& value) {
		CUDA_CHECK_THROW(
			cudaMemsetAsync(
				workspace.bitfield,
				value,
				workspace.n_bitfield_elements * sizeof(uint8_t),
				stream
			)
		);
	}

	// instance function to return the number of elements in this grid
	inline uint32_t NRC_HOST_DEVICE get_n_total_elements() const {
		return workspace.n_total_elements;
	}

	inline uint32_t NRC_HOST_DEVICE get_n_bitfield_elements() const {
		return workspace.n_bitfield_elements;
	}
	
	// get the size of the grid at a given level
	inline NRC_HOST_DEVICE float get_level_size(const int& level) const {
		return (float)(1 << level);
	}

	// gets the voxel size at a given level
	inline NRC_HOST_DEVICE float get_voxel_size(const int& level) const {
		return get_level_size(level) / resolution_f;
	}
	
	// returns the index of the voxel containing the point
	// assumes that xyz is in [0, 1]
	inline NRC_HOST_DEVICE uint32_t get_voxel_morton_index(
		const int& level,
		const float& x, const float& y, const float& z
	) const {
		// scale factor for each coordinate, based on the level
		const float s = 1.0f / get_level_size(level);
		
		const uint32_t x_i = (x * s + 0.5f) * resolution_f;
		const uint32_t y_i = (y * s + 0.5f) * resolution_f;
		const uint32_t z_i = (z * s + 0.5f) * resolution_f;
		
		// using morton code (Z-order curve), from Müller, et al. 2022 (page 15, "Occupancy Grids")
		return tcnn::morton3D(x_i, y_i, z_i);
	}

	// assigns x, y, z from the morton index
	// returns just the x,y,z indices from their respective axes
	inline NRC_HOST_DEVICE uint32_t get_voxel_xyz_index_from_morton_index(
		const uint32_t& morton_index,
		uint32_t& x, uint32_t& y, uint32_t& z
	) const {
		// scale factor for each coordinate, based on the level
		// converts [0, 1] -> [0, resolution] for the current level
		x = tcnn::morton3D_invert(morton_index >> 0);
		y = tcnn::morton3D_invert(morton_index >> 1);
		z = tcnn::morton3D_invert(morton_index >> 2);
	}

	// checks if the grid is occupied at a morton index of the given level
	inline NRC_HOST_DEVICE bool is_occupied_at(
		const int& level,
		const uint32_t& byte_idx
	) const {
		const uint8_t bitmask = (uint8_t)1 << (level % 8);
		return workspace.bitfield[byte_idx] & bitmask;
	}

	// checks if the grid is occupied at a position in the given level
	inline NRC_HOST_DEVICE bool is_occupied_at(
		const int& level,
		const float& x, const float& y, const float& z
	) const {
		const uint32_t byte_idx = get_voxel_morton_index(level, x, y, z) / 8;
		return is_occupied_at(level, byte_idx);
	}

	/* From Müller, et al. 2022
	 *
	 * Which one of the K grids is queried is determined by both the
	 * sample position xyz and the step size dt: among the grids covering xyz,
	 * the finest one with cell side-length larger than dt is queried.
	 */
	inline NRC_HOST_DEVICE int get_grid_level_at(
		const float& x, const float& y, const float& z,
		const float& dt_min
	) const {
		for (int k = 0; k < n_levels; ++k) {
			const float level_size = 1 << k;
			const float level_half_size = 0.5f * level_size;
			const float cell_size = level_size * inv_resolution_f;
			
			// might be able to eliminate the = in <= here
			const bool grid_k_covers_xyz = fabsf(x) <= level_half_size && fabsf(y) <= level_half_size && fabsf(z) <= level_half_size;
			if (grid_k_covers_xyz && cell_size > dt_min) {
				return k;
			}
		}
		// fallthrough - return largest grid
		return n_levels - 1;
	}

	// Stepping along a ray as in a Digital Differential Analyzer (DDA), get the distance to the next voxel
	// thank you NerfAcc
	inline NRC_HOST_DEVICE float get_t_to_next_voxel(
		const float& ray_pos_x, const float& ray_pos_y, const float& ray_pos_z,
		const float& ray_dir_x, const float& ray_dir_y, const float& ray_dir_z,
		const float& inv_dir_x, const float& inv_dir_y, const float& inv_dir_z,
		const float& dt_min,
		const int& grid_level
	) const {
		const float level_size= get_level_size(grid_level);
		const float i_level_size = 1.0f / level_size;

		// normalize xyz to [0, 1] for the current level
		const float x = (ray_pos_x * i_level_size + 0.5f) * resolution_f;
		const float y = (ray_pos_y * i_level_size + 0.5f) * resolution_f;
		const float z = (ray_pos_z * i_level_size + 0.5f) * resolution_f;
		
		// not really sure how this works, it's from NerfAcc.  It finds the t-space distance to the next voxel
		const float tx = ((floorf(0.5f * copysignf(1.0f, ray_dir_x) + x + 0.5f) - x) * inv_dir_x);
		const float ty = ((floorf(0.5f * copysignf(1.0f, ray_dir_y) + y + 0.5f) - y) * inv_dir_y);
		const float tz = ((floorf(0.5f * copysignf(1.0f, ray_dir_z) + z + 0.5f) - z) * inv_dir_z);

		return fmaxf(dt_min, fminf(fminf(tx, ty), tz)) / resolution_f * level_size;
	}

	// Gets the t-value required to step the ray to the next voxel
	// rewritten from NerfAcc without the while loop (see `advance_to_next_voxel`)
	inline NRC_HOST_DEVICE float get_dt_to_next_voxel(
		const float& ray_pos_x, const float& ray_pos_y, const float& ray_pos_z,
		const float& ray_dir_x, const float& ray_dir_y, const float& ray_dir_z,
		const float& inv_dir_x, const float& inv_dir_y, const float& inv_dir_z,
		const float& dt_min,
		const int& grid_level
	) const {
		const float t_target = get_t_to_next_voxel(
			ray_pos_x, ray_pos_y, ray_pos_z,
			ray_dir_x, ray_dir_y, ray_dir_z,
			inv_dir_x, inv_dir_y, inv_dir_z,
			dt_min,
			grid_level
		);
		
		return ceilf(t_target / dt_min) * dt_min;
	}

	// gets dt
	inline NRC_HOST_DEVICE float get_dt(
		const float& t,
		const float& cone_angle,
		const float& dt_min,
		const float& dt_max
	) const {
		return tcnn::clamp(t * cone_angle, dt_min, dt_max);
	}
};

NRC_NAMESPACE_END
