#pragma once

#include "../common.h"

TURBO_NAMESPACE_BEGIN

struct RayBatch {
    int size; // number of rays in this batch
    int stride; // stride between xyz components
    float* pos;
    float* dir;
    float* idir;
    float* t;
    float* transmittance;
    int* index;
    bool* active;
    bool* alive;
};

TURBO_NAMESPACE_END
