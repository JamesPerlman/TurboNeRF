#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
inline NRC_HOST_DEVICE uint32_t next_power_of_two(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
};

NRC_NAMESPACE_END
