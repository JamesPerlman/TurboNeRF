#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NetworkParamsWorkspace: Workspace {

    using Workspace::Workspace;

    float* params_fp;
    tcnn::network_precision_t* params_hp;
    tcnn::network_precision_t* gradients_hp;

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

        const uint32_t n_params = n_density_network_params + n_color_network_params;

        params_fp = allocate<float>(stream, n_params);
        params_hp = allocate<tcnn::network_precision_t>(stream, n_params);
        gradients_hp = allocate<tcnn::network_precision_t>(stream, n_params);

        density_network_params_fp = params_fp;
        density_network_params_hp = params_hp;
        density_network_gradients_hp = gradients_hp;

        color_network_params_fp = params_fp + n_density_network_params;
        color_network_params_hp = params_hp + n_density_network_params;
        color_network_gradients_hp = gradients_hp + n_density_network_params;
    }
};

NRC_NAMESPACE_END
