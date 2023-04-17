#pragma once

#include "ray-batch-coordinator.cuh"

TURBO_NAMESPACE_BEGIN

class LinearBufferRayBatchCoordinator : public RayBatchCoordinator {
    const int n_rays;
    const int start_idx;

public:
    LinearBufferRayBatchCoordinator(
        const int& n_rays,
        const int& start_idx = 0
    ) : n_rays(n_rays), start_idx(start_idx) {};

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