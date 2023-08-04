#pragma once

#include <tiny-cuda-nn/gpu_memory.h>

#include "../spatial-effect.cuh"
#include "../../math/transform4f.cuh"
#include "../../models/bounding-box.cuh"
#include "../../models/updatable-property.cuh"
#include "../../workspaces/workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct RepeaterEffectParams {
    BoundingBox source_bbox;
    BoundingBox extend_bbox;
    Transform4f transform;
};

class RepeaterEffect : public SpatialEffect<RepeaterEffectParams> {

public:
    RepeaterEffect(
        const BoundingBox& source_bbox,
        const BoundingBox& extend_bbox,
        const Transform4f& transform
    ) : SpatialEffect<RepeaterEffectParams>(
        { source_bbox, extend_bbox, transform }
    ) {};

    __host__ void apply(
        const cudaStream_t& stream,
        const uint32_t n_points,
        const uint32_t in_stride,
        const uint32_t out_stride,
        const float* in_xyz,
        float* out_xyz
    ) override;
};

TURBO_NAMESPACE_END
