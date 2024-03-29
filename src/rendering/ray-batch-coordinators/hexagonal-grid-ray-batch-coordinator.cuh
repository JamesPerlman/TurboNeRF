#pragma once

#include "../../common.h"
#include "../../models/camera.cuh"
#include "ray-batch-coordinator.cuh"

TURBO_NAMESPACE_BEGIN

class HexagonalGridRayBatchCoordinator : public RayBatchCoordinator {   
public:
    const int W, H, cw;
    const int2 grid_dims, grid_offset;

    HexagonalGridRayBatchCoordinator(
        const int2& grid_dims,
        const int2& grid_offset,
        const int& W,
        const int& H,
        const int& cw
    ) : grid_dims(grid_dims), grid_offset(grid_offset), W(W), H(H), cw(cw) {};

    void generate_rays(
        const Camera* camera,
        RayBatch& ray_batch,
        const cudaStream_t& stream = 0
    ) override;

    void copy_packed(
        const int& n_rays,
        const int2& output_size,
        const int& output_stride,
        float* rgba_in,
        float* rgba_out,
        const cudaStream_t& stream = 0
    ) override;
};

TURBO_NAMESPACE_END
