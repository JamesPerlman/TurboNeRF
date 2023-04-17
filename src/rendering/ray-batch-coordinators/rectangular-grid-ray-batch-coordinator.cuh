#pragma once

#include "../../common.h"
#include "ray-batch-coordinator.cuh"

TURBO_NAMESPACE_BEGIN

class RectangularGridRayBatchCoordinator : public RayBatchCoordinator {
public:
    int2 grid_offset; // offset of the grid in camera image space
    int2 grid_size; // size of the grid in pixels (camera image space)
    int2 grid_resolution; // number of points to sample in each dimension

    RectangularGridRayBatchCoordinator(
        const int2& grid_offset,
        const int2& grid_size,
        const int2& grid_resolution
    ) : grid_offset(grid_offset), grid_size(grid_size), grid_resolution(grid_resolution) {};

    void generate_rays(
        const Camera* camera,
        RayBatch& ray_batch,
        const cudaStream_t& stream = 0
    ) override;

    void copy_packed(
        const int& n_pixels,
        const int2& output_size,
        const int& output_stride,
        float* rgba_in,
        float* rgba_out,
        const cudaStream_t& stream = 0
    ) override;
};

TURBO_NAMESPACE_END
