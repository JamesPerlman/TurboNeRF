#pragma once

#include "repeater.cuh"
#include "../../math/tuple-math.cuh"

TURBO_NAMESPACE_BEGIN

__global__ void repeater_effect_kernel(
    const uint32_t n_points,
    const uint32_t in_stride,
    const uint32_t out_stride,
    const RepeaterEffectParams params,
    const bool* __restrict__ should_apply,
    const float* __restrict__ in_xyz,
    float* __restrict__ out_xyz
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    if (should_apply != nullptr && !should_apply[i]) return;

    const float3 in{
        in_xyz[0 * in_stride + i],
        in_xyz[1 * in_stride + i],
        in_xyz[2 * in_stride + i]
    };

    const float3 ext = params.source_bbox.size_xyz();
    const float3 min = params.source_bbox.min_xyz();
    const float3 p = (((in - min) % ext) + ext) % ext + min;

    

    out_xyz[0 * out_stride + i] = p.x;
    out_xyz[1 * out_stride + i] = p.y;
    out_xyz[2 * out_stride + i] = p.z;
}

__host__ void RepeaterEffect::apply(
        const cudaStream_t& stream,
        const uint32_t n_points,
        const uint32_t in_stride,
        const uint32_t out_stride,
        const bool* should_apply,
        const float* in_xyz,
        float* out_xyz
) const {
    repeater_effect_kernel<<<tcnn::n_blocks_linear(n_points), tcnn::n_threads_linear, 0, stream>>>(
        n_points,
        in_stride,
        out_stride,
        this->params,
        should_apply,
        in_xyz,
        out_xyz
    );
}

__host__ BoundingBox RepeaterEffect::get_bbox(
    const BoundingBox& bbox
) const {
    BoundingBox out;

    out.min_x = fminf(params.extend_bbox.min_x, bbox.min_x);
    out.min_y = fminf(params.extend_bbox.min_y, bbox.min_y);
    out.min_z = fminf(params.extend_bbox.min_z, bbox.min_z);

    out.max_x = fmaxf(params.extend_bbox.max_x, bbox.max_x);
    out.max_y = fmaxf(params.extend_bbox.max_y, bbox.max_y);
    out.max_z = fmaxf(params.extend_bbox.max_z, bbox.max_z);

    return out;
}

TURBO_NAMESPACE_END
