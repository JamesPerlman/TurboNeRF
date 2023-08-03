#pragma once

#include "../common.h"

TURBO_NAMESPACE_BEGIN

class ISpatialEffect {
    public:
    virtual ~ISpatialEffect() = default;
    virtual __host__ void apply(
        const cudaStream_t& stream,
        const uint32_t n_points,
        const uint32_t in_stride,
        const uint32_t out_stride,
        const float* __restrict__ in_xyz,
        float* __restrict__ out_xyz
    ) = 0;
};

template <typename Params>
class SpatialEffect: public ISpatialEffect {

    public:
    
    const Params& params;

    SpatialEffect(const Params& params) : params(params) {};

    virtual __host__ void apply(
        const cudaStream_t& stream,
        const uint32_t n_points,
        const uint32_t in_stride,
        const uint32_t out_stride,
        const float* __restrict__ in_xyz,
        float* __restrict__ out_xyz
    ) = 0;
};

TURBO_NAMESPACE_END
