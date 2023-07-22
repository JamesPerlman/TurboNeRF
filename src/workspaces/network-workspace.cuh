#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct NetworkWorkspace : Workspace {

    using Workspace::Workspace;

    // loss buffers
    float* ray_recon_loss;
    float* ray_dist_loss;

    // gradient calculation buffers
    float* grad_dLrecon_ddensity;
    float* grad_dLdist_ddensity;

    // these are intermediate buffers used by the distortion loss calculation
    float* sample_w_cs;
    float* sample_wm_cs;
    float* ray_dtw2_cs;
    float* ray_w_cs;
    float* ray_wm_cs;
    float* ray_wm_w_cs1_cs;
    float* ray_w_wm_cs1_cs;

    tcnn::network_precision_t* density_network_dL_doutput;
    tcnn::network_precision_t* color_network_dL_dinput;
    tcnn::network_precision_t* color_network_dL_doutput;

    uint32_t n_samples_per_batch = 0;
    uint32_t density_network_input_width = 0;
    uint32_t density_network_output_width = 0;
    uint32_t direction_encoding_input_width = 0;
    uint32_t direction_encoding_output_width = 0;
    uint32_t color_network_input_width = 0;
    uint32_t color_network_output_width = 0;

    void enlarge_if_needed(
        const cudaStream_t& stream,
        const uint32_t& n_samples_per_batch,
        const uint32_t& density_network_input_width,
        const uint32_t& density_network_output_width,
        const uint32_t& direction_encoding_input_width,
        const uint32_t& direction_encoding_output_width,
        const uint32_t& color_network_input_width,
        const uint32_t& color_network_output_width
    ) {

        const uint32_t batch_size = tcnn::next_multiple(n_samples_per_batch, tcnn::batch_size_granularity);
        
        if (
            batch_size == this->n_samples_per_batch &&
            density_network_input_width == this->density_network_input_width &&
            density_network_output_width == this->density_network_output_width &&
            direction_encoding_input_width == this->direction_encoding_input_width &&
            direction_encoding_output_width == this->direction_encoding_output_width &&
            color_network_input_width == this->color_network_input_width &&
            color_network_output_width == this->color_network_output_width
        ) {
            return;
        }
        
        free_allocations();

        this->n_samples_per_batch = n_samples_per_batch;
        this->density_network_input_width = density_network_input_width;
        this->density_network_output_width = density_network_output_width;
        this->direction_encoding_input_width = direction_encoding_input_width;
        this->direction_encoding_output_width = direction_encoding_output_width;
        this->color_network_input_width = color_network_input_width;
        this->color_network_output_width = color_network_output_width;
      
        ray_recon_loss = allocate<float>(stream, 4 * batch_size);
        ray_dist_loss = allocate<float>(stream, batch_size);

        grad_dLrecon_ddensity = allocate<float>(stream, batch_size);
        grad_dLdist_ddensity = allocate<float>(stream, batch_size);

        sample_w_cs = allocate<float>(stream, batch_size);
        sample_wm_cs = allocate<float>(stream, batch_size);
        ray_dtw2_cs = allocate<float>(stream, batch_size);
        ray_w_cs = allocate<float>(stream, batch_size);
        ray_wm_cs = allocate<float>(stream, batch_size);
        ray_wm_w_cs1_cs = allocate<float>(stream, batch_size);
        ray_w_wm_cs1_cs = allocate<float>(stream, batch_size);

        density_network_dL_doutput = allocate<tcnn::network_precision_t>(stream, density_network_output_width * batch_size);
        color_network_dL_doutput = allocate<tcnn::network_precision_t>(stream, color_network_output_width * batch_size);
        color_network_dL_dinput = allocate<tcnn::network_precision_t>(stream, color_network_input_width * batch_size);
    }

    // override free_allocations
    void free_allocations() override {
        
        n_samples_per_batch = 0;
        density_network_input_width = 0;
        density_network_output_width = 0;
        direction_encoding_input_width = 0;
        direction_encoding_output_width = 0;
        color_network_input_width = 0;
        color_network_output_width = 0;
        
        Workspace::free_allocations();
    }
};

TURBO_NAMESPACE_END
