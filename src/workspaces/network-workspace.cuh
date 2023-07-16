#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct NetworkWorkspace : Workspace {

    using Workspace::Workspace;

    uint32_t batch_size;

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

    float* density_network_dL_dinput;
    tcnn::network_precision_t* density_network_dL_doutput;
    tcnn::network_precision_t* color_network_dL_dinput;
    tcnn::network_precision_t* color_network_dL_doutput;

    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_samples_per_batch,
        const uint32_t& density_network_input_width,
        const uint32_t& density_network_output_width,
        const uint32_t& direction_encoding_input_width,
        const uint32_t& direction_encoding_output_width,
        const uint32_t& color_network_input_width,
        const uint32_t& color_network_output_width
    ) {
        free_allocations();

        batch_size = tcnn::next_multiple(n_samples_per_batch, tcnn::batch_size_granularity);

        density_network_dL_dinput = allocate<float>(stream, density_network_input_width * batch_size);
        color_network_dL_dinput = allocate<tcnn::network_precision_t>(stream, color_network_input_width * batch_size);

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
    }
};

TURBO_NAMESPACE_END
