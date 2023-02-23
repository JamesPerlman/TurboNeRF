/**
 * This class generates rays for the pixels of a hexagonal grid tile
 */

#pragma once

#include "../../common.h"
#include "ray-batch-coordinator.cuh"

NRC_NAMESPACE_BEGIN

class HexagonalTileRayBatchCoordinator : public RayBatchCoordinator {

public:

    struct HexagonTile {
        int n_rays;
        int x;
        int y;
        int H;
        int W;
        int cw;
        float fw;
        float fw1;
        float fw1_2;
        float fw1_sq_4;
        float fnp1_2;
    } tile;

    HexagonalTileRayBatchCoordinator(
        const int& n_rays,
        const int& W,
        const int& H,
        const int& cw,
        const int& x,
        const int& y
    ) {
        const float fw = cw;
        const float fw1 = (fw + 1);
        const float fw1_2 = 0.5f * fw1;
        const float fw1_sq_4 = 0.25f * fw1 * fw1;
        const float fnp1_2 = ((float)n_rays - 1.0f) / 2.0f;

        tile = {
            n_rays,
            x,
            y,
            H,
            W,
            cw,
            fw,
            fw1,
            fw1_2,
            fw1_sq_4,
            fnp1_2
        };
    };

    void generate_rays(
        const Camera* camera,
        const BoundingBox* bbox,
        RayBatch& ray_batch,
        const cudaStream_t& stream = 0
    ) override;

    void copy_packed(
        const int& n_pixels,
        const int& stride,
        const int2& output_size,
        float* rgba_in,
        float* rgba_out,
        const cudaStream_t& stream = 0
    ) override;
};

NRC_NAMESPACE_END
