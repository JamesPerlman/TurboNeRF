#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFNetworkWorkspace : Workspace {
    uint32_t batch_size;

    tcnn::network_precision_t* color_network_input;
    tcnn::network_precision_t* color_network_output;

    // gradient calculation buffers
    float* trans_buf;
    float* alpha_buf;
    float* weight_buf; // alpha * transmittance
    float* pxdiff_buf; // pixel channel differences
    float* ray_rgba; // accumulated ray colors from samples
    float* loss_buf;
    tcnn::network_precision_t* grad_buf;

	// buffers for normalized data
    float* normal_pos_batch;
    float* normal_dir_batch;
    float* normal_dt_batch;

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
        const uint32_t& color_network_output_width,
        const bool& for_training
    ) {
        free_allocations();

        batch_size = tcnn::next_multiple(n_samples_per_batch, tcnn::batch_size_granularity);
	
        if (for_training) {
            color_network_input = allocate<tcnn::network_precision_t>(stream, color_network_input_width * batch_size);
            color_network_output = allocate<tcnn::network_precision_t>(stream, color_network_output_width * batch_size);

            density_network_dL_dinput = allocate<float>(stream, density_network_input_width * batch_size);
            color_network_dL_dinput = allocate<tcnn::network_precision_t>(stream, color_network_input_width * batch_size);

            trans_buf = allocate<float>(stream, batch_size);
            alpha_buf = allocate<float>(stream, batch_size);
            weight_buf = allocate<float>(stream, batch_size);
            pxdiff_buf = allocate<float>(stream, 4 * batch_size);
            ray_rgba = allocate<float>(stream, 4 * batch_size);
            loss_buf = allocate<float>(stream, batch_size);
            grad_buf = allocate<tcnn::network_precision_t>(stream, color_network_output_width * batch_size);
        }

        normal_pos_batch = allocate<float>(stream, 3 * batch_size);
        normal_dir_batch = allocate<float>(stream, 3 * batch_size);
        normal_dt_batch = allocate<float>(stream, batch_size);
    }
};

NRC_NAMESPACE_END
