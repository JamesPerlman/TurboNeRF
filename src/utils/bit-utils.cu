#pragma once
#include "bit-utils.cuh"

TURBO_NAMESPACE_BEGIN

// counts the bits whose value is 1, given some buffer
// written by James Perlman, with a tiny bit of help from GPT-4!

__global__ void get_1s_per_uint32(
    const size_t n_elements,
    const uint32_t* __restrict__ buffer,
    uint8_t* __restrict__ counts
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const uint32_t value = buffer[idx];
    uint8_t count = static_cast<uint8_t>(__popc(value));
    counts[idx] = count;
}

TURBO_NAMESPACE_END
