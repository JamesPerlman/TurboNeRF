#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/camera.cuh"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct RenderingWorkspace: Workspace {

    using Workspace::Workspace;

    uint32_t batch_size;
    
    // compaction
    int* compact_idx;

    // rays
    bool* ray_alive;
    float* ray_origin[2];
    float* ray_dir[2];
    float* ray_tmax[2];
    float* ray_trans[2];

    // 2D ray index (x + y * width)
    int* ray_idx[2]; 
    
    // samples
    float* network_pos[2];
    float* network_dir[2];
    float* network_dt;
    int* net_compact_idx;
    int* sample_nerf_id;
    int* n_steps_total;

    // network buffers
    tcnn::network_precision_t* net_concat[2];
    tcnn::network_precision_t* net_output[2];

    // output buffers
    float* rgba;
    float* bg_rgba;

    uint32_t n_rays = 0;
    uint32_t n_nerfs = 0;
    size_t n_network_concat_elements = 0;
    size_t n_network_output_elements = 0;

    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_pixels,
        const uint32_t& n_elements_per_batch,
        const uint32_t& n_network_concat_elements,
        const uint32_t& n_network_output_elements
    ) {
        free_allocations();

        batch_size = tcnn::next_multiple(n_elements_per_batch, tcnn::batch_size_granularity);
        uint32_t n_output_pixel_elements = tcnn::next_multiple(4 * n_pixels, tcnn::batch_size_granularity);
        uint32_t n_rays = n_pixels;

        // compaction
        compact_idx     = allocate<int>(stream, n_rays);

        // rays
        ray_alive       = allocate<bool>(stream, n_rays); // no need to double buffer

        // double buffers

        ray_origin[0]   = allocate<float>(stream, 3 * n_rays);
        ray_origin[1]   = allocate<float>(stream, 3 * n_rays);
        
        ray_dir[0]      = allocate<float>(stream, 3 * n_rays);
        ray_dir[1]      = allocate<float>(stream, 3 * n_rays);
        
        ray_tmax[0]     = allocate<float>(stream, n_rays);
        ray_tmax[1]     = allocate<float>(stream, n_rays);

        ray_idx[0]      = allocate<int>(stream, n_rays);
        ray_idx[1]      = allocate<int>(stream, n_rays);

        ray_trans[0]    = allocate<float>(stream, n_rays);
        ray_trans[1]    = allocate<float>(stream, n_rays);

        // samples
        network_pos[0]  = allocate<float>(stream, 3 * batch_size);
        network_pos[1]  = allocate<float>(stream, 3 * batch_size);

        network_dir[0]  = allocate<float>(stream, 3 * batch_size);
        network_dir[1]  = allocate<float>(stream, 3 * batch_size);

        network_dt      = allocate<float>(stream, batch_size);

        net_compact_idx = allocate<int>(stream, batch_size);

        sample_nerf_id  = allocate<int>(stream, batch_size);

        n_steps_total   = allocate<int>(stream, batch_size);

        // network
        net_concat[0]   = allocate<tcnn::network_precision_t>(stream, n_network_concat_elements * batch_size);
        net_concat[1]   = allocate<tcnn::network_precision_t>(stream, n_network_concat_elements * batch_size);

        net_output[0]   = allocate<tcnn::network_precision_t>(stream, n_network_output_elements * batch_size);
        net_output[1]   = allocate<tcnn::network_precision_t>(stream, n_network_output_elements * batch_size);

        // output
        rgba            = allocate<float>(stream, n_output_pixel_elements);
        bg_rgba         = allocate<float>(stream, n_output_pixel_elements);

        this->n_rays = n_rays;
        this->n_network_concat_elements = n_network_concat_elements;
        this->n_network_output_elements = n_network_output_elements;
    };
};

TURBO_NAMESPACE_END