#pragma once

#include "../common.h"
#include "../models/bounding-box.cuh"

TURBO_NAMESPACE_BEGIN

class ISpatialEffect {
    public:
    virtual ~ISpatialEffect() = default;
    virtual __host__ void apply(
        const cudaStream_t& stream,
        const uint32_t n_points,
        const uint32_t in_stride,
        const uint32_t out_stride,
        const bool* should_apply,
        const float* __restrict__ in_xyz,
        float* __restrict__ out_xyz
    ) const = 0;

    virtual __host__ BoundingBox get_bbox(const BoundingBox& bbox) const = 0;
};

template <typename Params>
class SpatialEffect: public ISpatialEffect {

    public:
    
    const Params params;

    SpatialEffect(const Params& params) : params(params) {};

    virtual __host__ BoundingBox get_bbox(const BoundingBox& bbox) const {
        return bbox;
    };
};

TURBO_NAMESPACE_END
