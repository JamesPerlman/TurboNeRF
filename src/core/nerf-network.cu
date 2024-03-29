// This code was adapted from nerfstudio (Copyright 2022 The Nerfstudio Team. All rights reserved.)
// https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/instant_ngp_field.py
// Please see LICENSES/nerfstudio-project_nerfstudio.md for license details.

#include <json/json.hpp>
#include <math.h>
#include <tiny-cuda-nn/common.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "../utils/gpu-image.cuh"
#include "../utils/parallel-utils.cuh"
#include "../utils/training-kernels-fused.cuh"
#include "nerf-network.cuh"

using namespace tcnn;
using namespace turbo;
using json = nlohmann::json;


#if TCNN_HALF_PRECISION
    constexpr float LOSS_SCALE = 1024.0f;
#else
    constexpr float LOSS_SCALE = 1.0f;
#endif

void NerfNetwork::update_appearance_embedding_if_needed(
    const uint32_t& n_appearances,
    const uint32_t& appearance_embedding_dim
) {
    // These network configurations were adapted from nerfstudio

    // we need the density network in order to create the color network
    update_aabb_scale_if_needed(std::max(1, this->aabb_scale)); 

    // Create the Appearance Embedding
    appearance_embedding.reset(
        new Embedding<network_precision_t>(
            n_appearances,
            appearance_embedding_dim
        )
    );
    
    // Create the Direction Encoding
    json direction_encoding_config = {
        {"otype", "SphericalHarmonics"},
        {"degree", 4},
    };

    direction_encoding.reset(
        create_encoding<network_precision_t>(3, direction_encoding_config)
    );

    // Create the Color MLP
    const uint32_t n_dir_enc_dims = direction_encoding->padded_output_width();
    const uint32_t n_density_dims = density_network->padded_output_width();
    const uint32_t n_embedding_dims = appearance_embedding->padded_output_width();

    const uint32_t n_color_input_dims = n_dir_enc_dims + n_density_dims + n_embedding_dims;

    const json color_network_config = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "Sigmoid"},
        {"n_neurons", 64},
        {"n_hidden_layers", 2},
        {"n_input_dims", n_color_input_dims},
        {"n_output_dims", 3},
    };

    color_network.reset(
        create_network<network_precision_t>(color_network_config)
    );
}

// this creates (or recreates) the density network if needed
void NerfNetwork::update_aabb_scale_if_needed(const int& aabb_scale) {
    // there is no way to change the aabb_scale after the network is created
    // so as a workaround we just recreate the network if the aabb_scale changes
    // this will allow us to train multiple nerfs simultaneously using the same network

    if (this->aabb_scale == aabb_scale) {
        return;
    }

    // These values are from the Instant-NGP paper, page 4. "Multiresolution Hash Encoding"
    double N_min = 16.0;
    double N_max = 2048.0 * (double)aabb_scale;
    double n_levels = 16.0;

    double b = exp((log(N_max) - log(N_min)) / (n_levels - 1.0));

    // Create the Density MLP

    json density_encoding_config = {
        {"otype", "HashGrid"},
        {"n_levels", n_levels},
        {"n_features_per_level", 2},
        {"log2_hashmap_size", 19},
        {"base_resolution", N_min},
        {"per_level_scale", b},
        // We can use interpolation: Smoothstep here if needed.  See Müller et al (instant-NGP paper, page 13 "Smooth Interpolation")
        {"interpolation", "Linear"},
    };

    json density_network_config = {
        {"otype", "FullyFusedMLP"},
        {"activation", "ReLU"},
        {"output_activation", "None"},
        {"n_neurons", 64},
        {"n_hidden_layers", 1},
    };

    density_network.reset(
        new NetworkWithInputEncoding<network_precision_t>(
            3,  // input dims
            16, // output dims
            density_encoding_config,
            density_network_config
        )
    );

    this->aabb_scale = aabb_scale;
}

// initialize params and gradients for the networks
void NerfNetwork::update_params_if_needed(const cudaStream_t& stream, NetworkParamsWorkspace& params_ws) {

    _can_train = true;

    // initialize params
    if (params_ws.n_total_params == density_network->n_params() + color_network->n_params() + appearance_embedding->n_params()) {
        // params are already initialized
        return;
    }
    
    size_t rng_seed = 72791;
    pcg32 rng(rng_seed);

    params_ws.enlarge(
        stream,
        density_network->n_params(),
        color_network->n_params(),
        appearance_embedding->n_params()
    );
    
    density_network->initialize_params(rng, params_ws.density_network_params_fp);
    color_network->initialize_params(rng, params_ws.color_network_params_fp);
    appearance_embedding->initialize_params(rng, params_ws.appearance_embedding_params_fp);


    // initialize_params only initializes full precision params, need to copy to half precision

    copy_and_cast<network_precision_t, float>(
        stream,
        density_network->n_params(),
        params_ws.density_network_params_hp,
        params_ws.density_network_params_fp
    );

    copy_and_cast<network_precision_t, float>(
        stream,
        color_network->n_params(),
        params_ws.color_network_params_hp,
        params_ws.color_network_params_fp
    );

    copy_and_cast<network_precision_t, float>(
        stream,
        appearance_embedding->n_params(),
        params_ws.appearance_embedding_params_hp,
        params_ws.appearance_embedding_params_fp
    );
    
    json optimizer_config = {
        {"otype", "Adam"},
        {"learning_rate", 1e-2},
        {"epsilon", 1e-15},
        {"l2_reg", 1e-6}
    };

    optimizer.reset(
        new NGPAdamOptimizer<network_precision_t>(optimizer_config)
    );

    size_t n_params = density_network->n_params() + color_network->n_params() + appearance_embedding->n_params();
    uint32_t n_grid_params = density_network->encoding()->n_params();
    uint32_t n_network_params = color_network->n_params() + density_network->n_params();
    optimizer->allocate(n_params, {{n_grid_params, 1}, {n_network_params, 1}});
}

void NerfNetwork::free_device_memory() {
    workspace.free_allocations();
}

void NerfNetwork::free_training_data() {
    // TODO: free gradients, optimizer
    _can_train = false;
}

void NerfNetwork::set_params(NetworkParamsWorkspace& params_ws) {
    // assign params pointers

    density_network->set_params(
        params_ws.density_network_params_hp,
        params_ws.density_network_params_hp,
        params_ws.density_network_gradients_hp
    );

    color_network->set_params(
        params_ws.color_network_params_hp,
        params_ws.color_network_params_hp,
        params_ws.color_network_gradients_hp
    );

    appearance_embedding->set_params(
        params_ws.appearance_embedding_params_hp,
        params_ws.appearance_embedding_params_hp,
        params_ws.appearance_embedding_gradients_hp
    );
}

float NerfNetwork::train(
    const cudaStream_t& stream,
    NetworkParamsWorkspace& params_ws,
    const uint32_t& step,
    const uint32_t& batch_size,
    const uint32_t& n_rays,
    const uint32_t& n_samples,
    const uint32_t& n_rays_per_image,
    const int& aabb_scale,
    const float* random_rgb,
    uint32_t* ray_steps,
    uint32_t* ray_offset,
    uint32_t* appearance_id_batch,
    float* t_batch,
    float* pos_batch,
    float* dir_batch,
    float* dt_batch,
    float* m_norm_batch,
    float* dt_norm_batch,
    float* target_rgba,
    network_precision_t* concat_buffer,
    network_precision_t* output_buffer,
    const Settings& settings
) {

    update_aabb_scale_if_needed(aabb_scale);

    update_params_if_needed(stream, params_ws);

    set_params(params_ws);

    enlarge_workspace_if_needed(stream, batch_size);

    optimizer->set_learning_rate(1e-2f / (1.0f + NeRFConstants::learning_rate_decay * (float)step));

    // Forward
    auto fwd_ctx = forward(
        stream,
        batch_size,
        n_rays,
        n_samples,
        appearance_id_batch,
        pos_batch,
        dir_batch,
        concat_buffer,
        output_buffer
    );

    // custom kernels
    float distortion_loss = 0.0f;
    if (settings.use_distortion_loss) {
        distortion_loss = mipNeRF360_distortion_loss_forward_backward(
            stream,
            batch_size,
            n_rays,
            ray_steps,
            ray_offset,
            m_norm_batch,
            dt_norm_batch,
            concat_buffer
        );
    }

    float reconstruction_loss = fused_reconstruction_loss_forward_backward(
        stream,
        batch_size,
        n_rays,
        n_samples,
        ray_steps,
        ray_offset,
        random_rgb,
        target_rgba,
        pos_batch,
        dir_batch,
        dt_batch,
        concat_buffer,
        output_buffer
    );

    backward(
        stream,
        fwd_ctx,
        batch_size,
        n_rays,
        n_samples,
        n_rays_per_image,
        t_batch,
        settings
    );

    // Optimizer
    optimizer_step(stream, params_ws);

    return distortion_loss + reconstruction_loss;
}

void NerfNetwork::inference(
    const cudaStream_t& stream,
    NetworkParamsWorkspace& params_ws,
    const uint32_t& n_samples,
    const uint32_t& batch_size,
    const int& aabb_scale,
    uint32_t* appearance_id_batch,
    float* pos_batch,
    float* dir_batch,
    // density network output must have space available for (color_network->input_width() * batch_size) elements of type network_precision_t
    network_precision_t* concat_buffer,
    // color network output must have space available for (color_network->padded_output_width() * batch_size) elements of type network_precision_t
    network_precision_t* output_buffer,
    // if this flag is false, we only run inference on the density network
    const bool& use_color_network
) {
    update_aabb_scale_if_needed(aabb_scale);

    set_params(params_ws);
    
    // Inference (density network)
    GPUMatrixDynamic density_network_input_matrix(
        pos_batch,
        density_network->input_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    GPUMatrixDynamic density_network_output_matrix(
        concat_buffer,
        density_network->padded_output_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    density_network->inference_mixed_precision(
        stream,
        density_network_input_matrix,
        density_network_output_matrix
    );

    if (use_color_network) {
        // Inference (direction encoding)
        network_precision_t* direction_encoding_output = concat_buffer + density_network->padded_output_width() * batch_size;

        GPUMatrixDynamic direction_encoding_input_matrix(
            dir_batch,
            direction_encoding->input_width(),
            batch_size,
            MatrixLayout::RowMajor
        );

        GPUMatrixDynamic direction_encoding_output_matrix(
            direction_encoding_output,
            direction_encoding->padded_output_width(),
            batch_size,
            MatrixLayout::RowMajor
        );

        direction_encoding->inference_mixed_precision(
            stream,
            direction_encoding_input_matrix,
            direction_encoding_output_matrix
        );

        // Inference (appearance embedding)
        network_precision_t* appearance_embedding_output = direction_encoding_output + direction_encoding->padded_output_width() * batch_size;

        GPUMatrixDynamic appearance_embedding_input_matrix(
            appearance_id_batch,
            appearance_embedding->input_width(),
            batch_size,
            MatrixLayout::RowMajor
        );

        GPUMatrixDynamic appearance_embedding_output_matrix(
            appearance_embedding_output,
            appearance_embedding->padded_output_width(),
            batch_size,
            MatrixLayout::RowMajor
        );

        appearance_embedding->inference_mixed_precision(
            stream,
            n_samples,
            appearance_embedding_input_matrix,
            appearance_embedding_output_matrix
        );

        // Inference (color network)
        GPUMatrixDynamic color_network_input_matrix(
            concat_buffer,
            color_network->input_width(),
            batch_size,
            MatrixLayout::RowMajor
        );

        GPUMatrixDynamic color_network_output_matrix(
            output_buffer,
            color_network->padded_output_width(),
            batch_size,
            MatrixLayout::RowMajor
        );

        color_network->inference_mixed_precision(
            stream,
            color_network_input_matrix,
            color_network_output_matrix
        );
    }
}

std::unique_ptr<NerfNetwork::ForwardContext> NerfNetwork::forward(
    const cudaStream_t& stream,
    const uint32_t& batch_size,
    const uint32_t& n_rays,
    const uint32_t& n_samples,
    uint32_t* appearance_id_batch,
    float* pos_batch,
    float* dir_batch,
    network_precision_t* concat_buffer,
    network_precision_t* output_buffer
) {
    auto fwd_ctx = std::make_unique<ForwardContext>();

    // Forward pass on density network (with multiresolution hash encoding built in!)

    fwd_ctx->density_network_input_matrix = GPUMatrixDynamic(
        pos_batch,                              // density network takes the sample positions as input
        density_network->input_width(),         // rows
        batch_size,                             // cols
        MatrixLayout::RowMajor
    );

    // Here we make the output of the density network a pointer to the first half of the color network's input buffer.
    fwd_ctx->density_network_output_matrix = GPUMatrixDynamic(
        concat_buffer,                          // density network output = color network input
        density_network->output_width(),        // rows
        batch_size,                             // cols
        MatrixLayout::RowMajor
    );

    fwd_ctx->density_ctx = density_network->forward(
        stream,
        fwd_ctx->density_network_input_matrix,
        &fwd_ctx->density_network_output_matrix,
        false,
        true // prepare_input_gradients must be `true` otherwise backward() fails (forward->dy_dx is not defined)
    );

    // Encode directions (dir_batch)
    // Direction encoding gets concatenated with density_network_output (which will just be the second half of concat_buffer)

    network_precision_t* direction_encoding_output = concat_buffer + density_network->padded_output_width() * batch_size;

    fwd_ctx->direction_encoding_input_matrix = GPUMatrixDynamic(
        dir_batch,                                  // pointer to source data
        direction_encoding->input_width(),          // rows
        batch_size,                                 // cols
        MatrixLayout::RowMajor
    );

    fwd_ctx->direction_encoding_output_matrix = GPUMatrixDynamic(
        direction_encoding_output,                  // pointer to destination data
        direction_encoding->padded_output_width(),  // rows
        batch_size,                                 // cols
        MatrixLayout::RowMajor
    );

    direction_encoding->forward(
        stream,
        fwd_ctx->direction_encoding_input_matrix,
        &fwd_ctx->direction_encoding_output_matrix
    );

    // Apply appearance embedding (appearance_id_batch)

    network_precision_t* appearance_embedding_output = direction_encoding_output + direction_encoding->padded_output_width() * batch_size;

    fwd_ctx->appearance_embedding_input_matrix = GPUMatrixDynamic(
        appearance_id_batch,
        appearance_embedding->input_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    fwd_ctx->appearance_embedding_output_matrix = GPUMatrixDynamic(
        appearance_embedding_output,
        appearance_embedding->padded_output_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    appearance_embedding->forward(
        stream,
        n_samples,
        fwd_ctx->appearance_embedding_input_matrix,
        &fwd_ctx->appearance_embedding_output_matrix,
        false
    );

    // Perform the forward pass on the color network

    fwd_ctx->color_network_input_matrix = GPUMatrixDynamic(
        concat_buffer,                          // pointer to source data
        color_network->input_width(),           // matrix rows
        batch_size,                             // matrix columns
        MatrixLayout::RowMajor
    );

    fwd_ctx->color_network_output_matrix = GPUMatrixDynamic(
        output_buffer,                          // pointer to destination data
        color_network->padded_output_width(),   // matrix rows
        batch_size,                             // matrix columns
        MatrixLayout::RowMajor
    );

    fwd_ctx->color_ctx = color_network->forward(
        stream,
        fwd_ctx->color_network_input_matrix,
        &fwd_ctx->color_network_output_matrix,
        false,
        true // prepare_input_gradients
    );

    return fwd_ctx;
}

float NerfNetwork::fused_reconstruction_loss_forward_backward(
    const cudaStream_t& stream,
    const uint32_t& batch_size,
    const uint32_t& n_rays,
    const uint32_t& n_samples,

    const uint32_t* ray_steps,
    const uint32_t* ray_offset,
    const float* random_rgb,
    float* target_rgba,
    float* pos_batch,
    float* dir_batch,
    float* dt_batch,

    network_precision_t* concat_buffer,
    network_precision_t* output_buffer
) {
    // zero out previous gradients
    CUDA_CHECK_THROW(cudaMemsetAsync(workspace.grad_dLrecon_ddensity, 0, batch_size * sizeof(float), stream));
    CUDA_CHECK_THROW(
        cudaMemsetAsync(
            workspace.color_network_dL_doutput,
            0,
            color_network->padded_output_width() * batch_size * sizeof(network_precision_t),
            stream
        )
    );

    // perform custom fused forward/backward operations
    fused_reconstruction_forward_backward_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        batch_size,
        1.0f / (float)n_rays,
        LOSS_SCALE,
        ray_steps,
        ray_offset,
        random_rgb,
        target_rgba,
        concat_buffer,
        output_buffer,
        dt_batch,
        workspace.ray_recon_loss,
        workspace.grad_dLrecon_ddensity,
        workspace.color_network_dL_doutput
    );

    // return the sum of all losses
    thrust::device_ptr<float> loss_buffer_ptr(workspace.ray_recon_loss);

    return (1.0f / (float)n_rays) * thrust::reduce(
        MAKE_EXEC_POLICY(stream),
        loss_buffer_ptr,
        loss_buffer_ptr + 4 * batch_size,
        0.0f,
        thrust::plus<float>()
    );
}

float NerfNetwork::mipNeRF360_distortion_loss_forward_backward(
    const cudaStream_t& stream,
    const uint32_t& batch_size,
    const uint32_t& n_rays,
    const uint32_t* ray_steps,
    const uint32_t* ray_offset,
    const float* m_norm_batch,
    const float* dt_norm_batch,
    const tcnn::network_precision_t* concat_buffer
) {
    // zero out previous gradients
    CUDA_CHECK_THROW(cudaMemsetAsync(workspace.grad_dLdist_ddensity, 0, batch_size * sizeof(float), stream));

    mipNeRF360_distortion_loss_forward_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        batch_size,
        ray_steps,
        ray_offset,
        concat_buffer,
        m_norm_batch,
        dt_norm_batch,
        workspace.ray_dtw2_cs,
        workspace.ray_w_cs,
        workspace.ray_wm_cs,
        workspace.ray_wm_w_cs1_cs,
        workspace.ray_w_wm_cs1_cs,
        workspace.ray_dist_loss,
        workspace.sample_w_cs,
        workspace.sample_wm_cs
    );

    mipNeRF360_distortion_loss_backward_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        batch_size,
        ray_steps,
        ray_offset,
        concat_buffer,
        workspace.ray_dtw2_cs,
        workspace.ray_w_cs,
        workspace.ray_wm_cs,
        workspace.ray_wm_w_cs1_cs,
        workspace.ray_w_wm_cs1_cs,
        workspace.ray_dist_loss,
        m_norm_batch,
        dt_norm_batch,
        workspace.sample_w_cs,
        workspace.sample_wm_cs,
        workspace.grad_dLdist_ddensity
    );

    // return the sum of all loss values

    thrust::device_ptr<float> loss_buffer_ptr(workspace.ray_dist_loss);

    return (1.0 / (float)n_rays) * thrust::reduce(
        MAKE_EXEC_POLICY(stream),
        loss_buffer_ptr,
        loss_buffer_ptr + n_rays,
        0.0f,
        thrust::plus<float>()
    );
}

void NerfNetwork::backward(
    const cudaStream_t& stream,
    const std::unique_ptr<NerfNetwork::ForwardContext>& fwd_ctx,
    const uint32_t& batch_size,
    const uint32_t& n_rays,
    const uint32_t& n_samples,
    const uint32_t& n_rays_per_image,
    const float* t_batch,
    const NerfNetwork::Settings& settings
) {
    
    // Backpropagate through the color network

    // color network's dL_doutput was computed by the reconstruction backward kernel
    GPUMatrixDynamic color_network_dL_doutput_matrix(
        workspace.color_network_dL_doutput,
        color_network->padded_output_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    GPUMatrixDynamic color_network_dL_dinput_matrix(
        workspace.color_network_dL_dinput,
        color_network->input_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    color_network->backward(
        stream,
        *fwd_ctx->color_ctx,
        fwd_ctx->color_network_input_matrix,
        fwd_ctx->color_network_output_matrix,
        color_network_dL_doutput_matrix,
        &color_network_dL_dinput_matrix
    );

    // Backpropagate through the appearance embedding

    // appearance embedding's dL_doutput exists at the end part of the color network's dL_dinput
    uint32_t appearance_embedding_offset = (color_network->input_width() - appearance_embedding->padded_output_width()) * batch_size;
    GPUMatrixDynamic appearance_embedding_dL_doutput_matrix(
        workspace.color_network_dL_dinput + appearance_embedding_offset,
        appearance_embedding->padded_output_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    appearance_embedding->backward(
        stream,
        n_samples,
        n_rays_per_image,
        fwd_ctx->appearance_embedding_input_matrix,
        appearance_embedding_dL_doutput_matrix
    );

    // Backpropagate through the density network

    // density network's dL_doutput is the beginning part of the color network's dL_dinput
    GPUMatrixDynamic density_network_dL_doutput_matrix(
        color_network_dL_dinput_matrix.data(),
        density_network->padded_output_width(),
        batch_size,
        MatrixLayout::RowMajor
    );

    // We need to scale dL/ddensity and add it to dL/doutput before backpropagating
    copy_gradients_kernel<1, true><<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
        n_samples,
        batch_size,
        LOSS_SCALE,
        workspace.grad_dLrecon_ddensity,
        density_network_dL_doutput_matrix.data()
    );

    // We also may need to copy the distortion loss
    if (settings.use_distortion_loss) {
        copy_gradients_kernel<1, true><<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
            n_samples,
            batch_size,
            LOSS_SCALE,
            workspace.grad_dLdist_ddensity,
            density_network_dL_doutput_matrix.data()
        );
    }

    apply_gradient_distance_scaling_kernel<<<n_blocks_linear(n_samples), n_threads_linear, 0, stream>>>(
        n_samples,
        t_batch,
        density_network_dL_doutput_matrix.data()
    );

    density_network->backward(
        stream,
        *fwd_ctx->density_ctx,
        fwd_ctx->density_network_input_matrix,
        fwd_ctx->density_network_output_matrix,
        density_network_dL_doutput_matrix
    );

}

void NerfNetwork::optimizer_step(const cudaStream_t& stream, NetworkParamsWorkspace& params_ws) {

    optimizer->step(
        stream,
        LOSS_SCALE,
        params_ws.params_fp,
        params_ws.params_hp,
        params_ws.gradients_hp
    );

}

// Only enlarge buffers needed for inference
void NerfNetwork::enlarge_workspace_if_needed(const cudaStream_t& stream, const uint32_t& batch_size) {
    workspace.enlarge_if_needed(
        stream,
        batch_size,
        density_network->input_width(),
        density_network->padded_output_width(),
        direction_encoding->input_width(),
        direction_encoding->padded_output_width(),
        color_network->input_width(),
        color_network->padded_output_width()
    );

    this->batch_size = batch_size;
}
