#pragma once
#include "../common.h"

NRC_NAMESPACE_BEGIN

struct Rays {
    float* ori_x;
    float* ori_y;
    float* ori_z;

    float* dir_x;
    float* dir_y;
    float* dir_z;

    float* t;

    uint32_t* steps;
    uint32_t* cum_steps;
}

struct RenderingRays: Rays {
    bool* alive;
    bool* active;
    uint32_t* pix_x;
    uint32_t* pix_y;
}
