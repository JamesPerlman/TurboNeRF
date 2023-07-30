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
    BoundingBox extension_bbox;
    Transform4f transform;
};

class RepeaterEffect : public SpatialEffect<RepeaterEffectParams> {
    public:
    __device__ void apply(
        const RepeaterEffectParams& params,
        const float& in_x, const float& in_y, const float& in_z,
        float& out_x, float& out_y, float& out_z
    ) override {
        out_x = fmodf(in_x, params.source_bbox.size_x());
        out_y = fmodf(in_y, params.source_bbox.size_y());
        out_z = fmodf(in_z, params.source_bbox.size_z());
    }
};

TURBO_NAMESPACE_END
