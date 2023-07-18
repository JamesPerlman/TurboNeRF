#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

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

    float* appearance_embedding_params_fp;
    tcnn::network_precision_t* appearance_embedding_params_hp;
    tcnn::network_precision_t* appearance_embedding_gradients_hp;

    uint32_t n_density_params;
    uint32_t n_color_params;
    uint32_t n_appearance_embedding_params;

    uint32_t n_total_params;

    // TODO: inference-only mode (no gradients)
    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_density_network_params,
        const uint32_t& n_color_network_params,
        const uint32_t& n_appearance_embedding_params
    ) {
        free_allocations();

        n_density_params = n_density_network_params;
        n_color_params = n_color_network_params;
        n_total_params = n_density_params + n_color_params + n_appearance_embedding_params;

        params_fp = allocate<float>(stream, n_total_params);
        params_hp = allocate<tcnn::network_precision_t>(stream, n_total_params);
        gradients_hp = allocate<tcnn::network_precision_t>(stream, n_total_params);

        density_network_params_fp = params_fp;
        density_network_params_hp = params_hp;
        density_network_gradients_hp = gradients_hp;

        color_network_params_fp = params_fp + n_density_params;
        color_network_params_hp = params_hp + n_density_params;
        color_network_gradients_hp = gradients_hp + n_density_params;

        appearance_embedding_params_fp = color_network_params_fp + n_color_params;
        appearance_embedding_params_hp = color_network_params_hp + n_color_params;
        appearance_embedding_gradients_hp = color_network_gradients_hp + n_color_params;
    }

    // override free_allocations
    void free_allocations() override {

        n_density_params = 0;
        n_color_params = 0;
        n_total_params = 0;
        
        Workspace::free_allocations();
    }
};

TURBO_NAMESPACE_END
