#include "occupancy-grid-kernels.cuh"

/**
 * This entire file is a direct implementation of page 15, "Updating the occupancy grids" of this paper:
 * 
 * Müller, Thomas, et al. "Instant neural graphics primitives with a multiresolution hash encoding."
 * *ACM Trans. Graph.*, 41(4), 102:1-102:15 - https://doi.org/10.1145/3528223.3530127  
 */

using namespace tcnn;

NRC_NAMESPACE_BEGIN

// occupancy cell values first get decayed by a factor (default 0.95) every update
__global__ void decay_occupancy_grid_values_kernel(
    const uint32_t n_cells_per_level,
    const uint32_t n_levels,
    const float factor,
    float* __restrict__ grid_density
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells_per_level) {
        return;
    }

    float* d = grid_density + idx;

    #pragma unroll
    for (int i = 0; i < n_levels; ++i) {
        *d *= factor;
        d += n_cells_per_level;
    }
}

// generate points in the grid for sampling the sigma network
// the network_pos values are in the range [0, 1]
__global__ void generate_grid_cell_network_sample_points_kernel(
    const uint32_t n_cells,
    const uint32_t batch_size,
    const uint32_t start_idx,
    const CascadedOccupancyGrid* __restrict__ grid,
    const int level,
    const float inv_aabb_size,
    const float* __restrict__ random_float,
    float* __restrict__ sample_pos
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) {
        return;
    }
    
    const uint32_t idx = i + start_idx;

    const float level_size = grid->get_level_size(level);
    const float voxel_size = level_size * grid->inv_resolution_f;
    
    const uint32_t i_offset_0 = i;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;

    // get xyz positions of the grid cell according to the morton code index
    float vx, vy, vz;
    grid->get_voxel_xyz_from_morton_index(idx, level, vx, vy, vz);

    // set each dimension of sample_pos to the corner of the grid cell + a random offset
    // this x,y,z is in world coordinates
    const float x = vx + random_float[i_offset_0] * voxel_size;
    const float y = vy + random_float[i_offset_1] * voxel_size;
    const float z = vz + random_float[i_offset_2] * voxel_size;

    // Normalize the sample position to the range [0, 1] (for the network)
    sample_pos[i_offset_0] = (x * inv_aabb_size) + 0.5f;
    sample_pos[i_offset_1] = (y * inv_aabb_size) + 0.5f;
    sample_pos[i_offset_2] = (z * inv_aabb_size) + 0.5f;

}

// occupancy cell values are updated to the maximum of the current value and a newly sampled density value
__global__ void update_occupancy_with_density_kernel(
    const uint32_t n_samples,
    const uint32_t start_idx,
    const uint32_t level,
    const float selection_threshold,
    const float* __restrict__ random_float,
    const tcnn::network_precision_t* __restrict__ network_density,
    CascadedOccupancyGrid* __restrict__ grid
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) {
        return;
    }

    const uint32_t idx = i + start_idx;

    // (selection_threshold * 100)% of cells are sampled randomly, and half of the rest are sampled based on the current occupancy
    if (selection_threshold < random_float[i]) {
        if (selection_threshold < 0.5f * random_float[i]) {
            if (!grid->is_occupied_at(level, idx)) {
                return;
            }
        } else {
            return;
        }
    }

    grid->update_sigma_at(level, idx, (float)network_sigma[i]);
}

// occupancy bits are updated by thresholding each cell's density, default = 0.01 * 1024 / sqrt(3)
__global__ void update_occupancy_grid_bits_kernel(
    const uint32_t n_cells_per_level,
    const int n_levels,
    const float threshold,
    CascadedOccupancyGrid* __restrict__ grid,
    const float* __restrict__ grid_density,
    uint8_t* __restrict__ grid_bits
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells_per_level) {
        return;
    }

    uint8_t cell_bits = 0b00000000;

    #pragma unroll
    for (int level = 0; level < n_levels; ++level) {
        const uint32_t density_idx = level * n_cells_per_level + idx;
        
        // get "is threshold exceeded?" as a bit
        uint8_t b = grid_density[density_idx] > threshold ? 1 : 0;
        cell_bits |= b << level;
    }

    grid_bits[idx] = cell_bits;
}

NRC_NAMESPACE_END
