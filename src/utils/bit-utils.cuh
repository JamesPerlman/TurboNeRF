#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common.h"

TURBO_NAMESPACE_BEGIN

// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
template <typename T>
inline NRC_HOST_DEVICE T next_power_of_two(T v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
};

TURBO_NAMESPACE_END
