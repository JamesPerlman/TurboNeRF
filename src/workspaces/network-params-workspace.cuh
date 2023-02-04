#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NetworkParamsWorkspace: Workspace {
    
    float* density_network_params_fp;
    tcnn::network_precision_t* density_network_params_hp;
    tcnn::network_precision_t* density_network_gradients_hp;

    float* color_network_params_fp;
    tcnn::network_precision_t* color_network_params_hp;
    tcnn::network_precision_t* color_network_gradients_hp;

    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_density_network_params,
        const uint32_t& n_color_network_params
    ) {
        free_allocations();

        density_network_params_fp = allocate<float>(stream, n_density_network_params);
        density_network_params_hp = allocate<tcnn::network_precision_t>(stream, n_density_network_params);
        density_network_gradients_hp = allocate<tcnn::network_precision_t>(stream, n_density_network_params);

        color_network_params_fp = allocate<float>(stream, n_color_network_params);
        color_network_params_hp = allocate<tcnn::network_precision_t>(stream, n_color_network_params);
        color_network_gradients_hp = allocate<tcnn::network_precision_t>(stream, n_color_network_params);
    }
};

NRC_NAMESPACE_END
