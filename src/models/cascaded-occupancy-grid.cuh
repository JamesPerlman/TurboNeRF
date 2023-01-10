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
private:
	uint32_t n_levels;
	float resolution_f;
	float inv_resolution_f;
	uint32_t resolution_i;

	CascadedOccupancyGridWorkspace workspace;

public:
	// level is the power of two domain size (K from pg. 15 of Müller, et al. 2022)
	// grid goes from [-2^(level - 1) + 0.5, 2^(level - 1) + 0.5] in each dimension
	CascadedOccupancyGrid(uint32_t n_levels, uint32_t resolution = 128)
		: n_levels(n_levels)
		, resolution_i(resolution)
		, resolution_f(resolution)
		, inv_resolution_f(1.0f / (float)resolution)
	{};

	CascadedOccupancyGrid() = default;

	// class function to calculate number of bytes needed to store the grid
	static inline NRC_HOST_DEVICE size_t get_n_total_elements(uint32_t n_levels, uint32_t grid_resolution = 128) {
		uint32_t side_length = next_power_of_two(grid_resolution);
		return (size_t)(n_levels * side_length * side_length * side_length);
	}
	
	// allocators/initializers
	uint8_t* initialize_bitfield(const cudaStream_t& stream) {
		workspace.enlarge_bitfield(stream, CascadedOccupancyGrid::get_n_total_elements(n_levels, resolution_i));
		return workspace.bitfield;
	}
	
	float* initialize_values(const cudaStream_t& stream) {
		workspace.enlarge_values(stream, CascadedOccupancyGrid::get_n_total_elements(n_levels, resolution_i));
		return workspace.values;
	}

	// pointer getters
	inline NRC_HOST_DEVICE uint8_t* get_bitfield() const {
		return workspace.bitfield;
	}

	inline NRC_HOST_DEVICE float* get_values() const {
		return workspace.values;
	}

	// instance function to return the number of elements in this grid
	inline uint32_t NRC_HOST_DEVICE get_n_total_elements() const {
		return workspace.n_total_elements;
	}

	inline uint32_t NRC_HOST_DEVICE get_n_bitfield_elements() const {
		return workspace.n_bitfield_elements;
	}

	// returns the total volume of the grid
	inline NRC_HOST_DEVICE uint32_t volume() const {
		return resolution_i * resolution_i * resolution_i;
	}

	// normalize a coordinate to [0, 1] within a grid at level k
	inline NRC_HOST_DEVICE float3 get_normalized_coordinates(const uint32_t& k, const float& x, const float& y, const float& z) const {
		float scale = 1 << k;
		return {
			x / scale + 0.5f,
			y / scale + 0.5f,
			z / scale + 0.5f
		};
	}
	
	// returns the index of the voxel containing the point
	inline NRC_HOST_DEVICE uint32_t get_voxel_morton_index(
		const uint32_t& level,
		const float& x, const float& y, const float& z
	) const {
		float scale = 1 << level;
		
		uint32_t x_i = (uint32_t)((x / scale + 0.5f) * resolution_f);
		uint32_t y_i = (uint32_t)((y / scale + 0.5f) * resolution_f);
		uint32_t z_i = (uint32_t)((z / scale + 0.5f) * resolution_f);
		
		// using morton code (Z-order curve), from Müller, et al. 2022 (page 15, "Occupancy Grids")
		return tcnn::morton3D(x_i, y_i, z_i);
	}
	
	// checks if the grid is occupied or not*
	inline NRC_HOST_DEVICE bool is_occupied_at(
		const uint8_t& level,
		const float& x, const float& y, const float& z
	) const {
		uint32_t byte_idx = get_voxel_morton_index(level, x, y, z) / 8;
		uint8_t bitmask = 1 << ((n_levels * byte_idx + level) % 8);
		return workspace.bitfield[byte_idx] & bitmask;
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
			float level_size = 1 << k;
			float level_extent = 0.5f * level_size;
			float cell_size = level_size * inv_resolution_f;
			
			// might be able to eliminate the = in <= here
			bool grid_k_covers_xyz = fabsf(x) <= level_extent && fabsf(y) <= level_extent && fabsf(z) <= level_extent;
			if (grid_k_covers_xyz && cell_size > dt_min) {
				return k;
			}
		}
	}

	// Stepping along a ray as in a Digital Differential Analyzer (DDA), get the distance to the next voxel
	// ray_pos is assumed to be normalized in [0, 1].  Output is in normalized space as well.
	// thank you NerfAcc
	inline NRC_HOST_DEVICE float get_t_to_next_voxel(
		const float& ray_pos_x, const float& ray_pos_y, const float& ray_pos_z,
		const float& ray_dir_x, const float& ray_dir_y, const float& ray_dir_z,
		const float& inv_dir_x, const float& inv_dir_y, const float& inv_dir_z
	) const {
		float x = ray_dir_x * resolution_f;
		float y = ray_dir_y * resolution_f;
		float z = ray_dir_z * resolution_f;

		float tx = ((floorf(0.5f * copysignf(1.0f, ray_dir_x) + x + 0.5f) - x) * inv_dir_x);
		float ty = ((floorf(0.5f * copysignf(1.0f, ray_dir_y) + y + 0.5f) - y) * inv_dir_y);
		float tz = ((floorf(0.5f * copysignf(1.0f, ray_dir_z) + z + 0.5f) - z) * inv_dir_z);

		return fmaxf(0.0f, fminf(fminf(tx, ty), tz)) / resolution_f;
	}

	// Gets the t-value required to step the ray to the next voxel
	// rewritten from NerfAcc without the while loop (see `advance_to_next_voxel`)
	inline NRC_HOST_DEVICE float get_t_advanced_to_next_voxel(
		const float& ray_pos_x, const float& ray_pos_y, const float& ray_pos_z,
		const float& ray_dir_x, const float& ray_dir_y, const float& ray_dir_z,
		const float& inv_dir_x, const float& inv_dir_y, const float& inv_dir_z,
		const float& t,
		const float& dt_min
	) const {
		float t_target = get_t_to_next_voxel(
			ray_pos_x, ray_pos_y, ray_pos_z,
			ray_dir_x, ray_dir_y, ray_dir_z,
			inv_dir_x, inv_dir_y, inv_dir_z
		);
		return t + ceilf(t_target / dt_min) * dt_min;
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