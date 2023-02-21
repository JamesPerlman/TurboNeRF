#pragma once

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct RayBatch {
    int size; // number of rays in this batch
    int stride; // stride between xyz components
    float* pos;
    float* dir;
    float* idir;
    float* t;
    float* transmittance;
    int* index;
    uint8_t* flags; // optimization to avoid multiple global memory accesses
};

enum class RayFlags: uint8_t {
    Alive   = 1 << 0,
    Active  = 1 << 1,
};

NRC_NAMESPACE_END
