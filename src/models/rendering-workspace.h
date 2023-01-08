#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "camera.cuh"

NRC_NAMESPACE_BEGIN

struct RenderingWorkspace {

    uint32_t batch_size;
    
    const Camera* camera;

    // rays
    bool* ray_alive;
    bool* ray_active;

    float* ray_origin;
    float* ray_dir;
    float* ray_t;
    
    uint32_t* ray_steps;
    uint32_t* ray_steps_cumulative;

    uint32_t* ray_x;
    uint32_t* ray_y;
    
    // samples
    float* sample_pos;
    float* sample_dir;
    float* sample_dt;

    // output buffers
    tcnn::network_precision_t* network_output;
    float* output_buffer;

    // samples
    void enlarge(const size_t& output_width, const size_t& output_height);

};

NRC_NAMESPACE_END