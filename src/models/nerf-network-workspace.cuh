#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFNetworkWorkspace : Workspace {
    uint32_t batch_size;

    // gradient calculation buffers
    float* trans_buf;
    float* weight_buf; // alpha * transmittance
    float* sigma_buf; // e^(sigma - 1) 
    float* pxdiff_buf; // pixel channel differences
    float* ray_rgba; // accumulated ray colors from samples
    float* loss_buf;
    float* grad_dL_dR;
    float* grad_dL_dweight;
    float* grad_dL_dsigma;
    tcnn::network_precision_t* grad_dL_dcolor;
    tcnn::network_precision_t* grad_dL_ddensity;


	// buffers for backpropagation
    float* density_network_dL_dinput;
    tcnn::network_precision_t* color_network_dL_dinput;

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

        trans_buf = allocate<float>(stream, batch_size);
        weight_buf = allocate<float>(stream, batch_size);
        sigma_buf = allocate<float>(stream, batch_size);
        pxdiff_buf = allocate<float>(stream, 4 * batch_size);
        ray_rgba = allocate<float>(stream, 4 * batch_size);
        loss_buf = allocate<float>(stream, 4 * batch_size);
       
        grad_dL_dR = allocate<float>(stream, batch_size);
        grad_dL_dweight = allocate<float>(stream, batch_size);
        grad_dL_dsigma = allocate<float>(stream, batch_size);
        grad_dL_dcolor = allocate<tcnn::network_precision_t>(stream, 3 * batch_size);
        grad_dL_ddensity = allocate<tcnn::network_precision_t>(stream, batch_size);
    }
};

NRC_NAMESPACE_END
