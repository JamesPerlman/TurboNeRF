#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct NetworkWorkspace : Workspace {

    using Workspace::Workspace;

    uint32_t batch_size;

    // gradient calculation buffers
    float* sigma_buf; // e^(sigma - 1) 
    float* alpha_buf;
    float* trans_buf;
    float* weight_buf; // alpha * transmittance
    float* pxdiff_buf; // pixel channel differences
    float* ray_rgba; // accumulated ray colors from samples
    float* loss_buf;
    float* grad_dL_dR;
    float* grad_dL_dweight;
    float* grad_dL_dsigma;
    float* grad_dL_dcolor;
    float* grad_dL_ddensity;
    float* grad_dL_dalpha;
    float* grad_dL_dtrans;

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

        sigma_buf = allocate<float>(stream, batch_size);
        alpha_buf = allocate<float>(stream, batch_size);
        trans_buf = allocate<float>(stream, batch_size);
        weight_buf = allocate<float>(stream, batch_size);
        pxdiff_buf = allocate<float>(stream, 4 * batch_size);
        ray_rgba = allocate<float>(stream, 4 * batch_size);
        loss_buf = allocate<float>(stream, 4 * batch_size);
       
        grad_dL_dR = allocate<float>(stream, 4 * batch_size);
        grad_dL_dweight = allocate<float>(stream, batch_size);
        grad_dL_dsigma = allocate<float>(stream, batch_size);
        grad_dL_dcolor = allocate<float>(stream, 3 * batch_size);
        grad_dL_ddensity = allocate<float>(stream, batch_size);
        grad_dL_dalpha = allocate<float>(stream, batch_size);
        grad_dL_dtrans = allocate<float>(stream, batch_size);

        density_network_dL_doutput = allocate<tcnn::network_precision_t>(stream, density_network_output_width * batch_size);
        color_network_dL_doutput = allocate<tcnn::network_precision_t>(stream, color_network_output_width * batch_size);
    }
};

TURBO_NAMESPACE_END
