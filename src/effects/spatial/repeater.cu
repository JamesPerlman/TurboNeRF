#pragma once

#include "repeater.cuh"
#include "../../math/tuple-math.cuh"

TURBO_NAMESPACE_BEGIN

__global__ void repeater_effect_kernel(
    const uint32_t n_points,
    const uint32_t in_stride,
    const uint32_t out_stride,
    const RepeaterEffectParams params,
    const float* __restrict__ in_xyz,
    float* __restrict__ out_xyz
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_points) return;

    const float3 in{
        in_xyz[0 * in_stride + i],
        in_xyz[1 * in_stride + i],
        in_xyz[2 * in_stride + i]
    };

    float3 sc = params.source_bbox.center();
    float3 ss = params.source_bbox.size_xyz();
    float3 sh = 0.5f * ss;

    out_xyz[0 * out_stride + i] = fmodf(in.x - sh.x - sc.x, ss.x) - sh.x;
    out_xyz[1 * out_stride + i] = fmodf(in.y - sh.y - sc.y, ss.y) - sh.y;
    out_xyz[2 * out_stride + i] = fmodf(in.z - sh.z - sc.z, ss.z) - sh.z;
}

__host__ void RepeaterEffect::apply(
        const cudaStream_t& stream,
        const uint32_t n_points,
        const uint32_t in_stride,
        const uint32_t out_stride,
        const float* in_xyz,
        float* out_xyz
) {
    repeater_effect_kernel<<<tcnn::n_blocks_linear(n_points), tcnn::n_threads_linear, 0, stream>>>(
        n_points,
        in_stride,
        out_stride,
        this->params,
        in_xyz,
        out_xyz
    );
}

TURBO_NAMESPACE_END
