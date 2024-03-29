/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   adam-optimizer.cuh
 *  @author Thomas Müller, NVIDIA
 *  @author Modified by James Perlman on 2023-02-15, copied from tiny-cuda-nn
 *  @brief  Implementation of the adam optimizer according to the Instant-NGP paper.
 *  With minor optimizations and no L2 regularization for hash grid parameters.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <json/json.hpp>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

TCNN_NAMESPACE_BEGIN

template <typename T>
__global__ void ngp_adam_step(
    const uint32_t n_elements,
    const uint32_t hash_grid_weights_final_index,
    const uint32_t network_weights_final_index,
    const float loss_scale,
    float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float l2_reg,
    float* __restrict__ weights_full_precision,
    T* __restrict__ weights,
    const T* __restrict__ gradients,
    float* __restrict__ first_moments,
    float* __restrict__ second_moments,
    uint32_t* __restrict__ param_steps
) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    float gradient = (float)gradients[i] / loss_scale;
    if (i < hash_grid_weights_final_index && gradient == 0) {
        return;
    }

    const float weight_fp = weights_full_precision[i];

    // L2 regularization only for network weights
    if (i >= hash_grid_weights_final_index && i < network_weights_final_index) {
        gradient += l2_reg * weight_fp;
    }

    const float gradient_sq = gradient * gradient;

    float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
    const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;

    // Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
    const uint32_t current_step = ++param_steps[i];
    learning_rate *= sqrtf(1 - powf(beta2, (float)current_step)) / (1 - powf(beta1, (float)current_step));

    const float effective_learning_rate = learning_rate / (sqrtf(second_moment) + epsilon);

    const float new_weight = weight_fp - effective_learning_rate * first_moment;

    weights_full_precision[i] = new_weight;
    weights[i] = (T)new_weight;
}

template <typename T>
class NGPAdamOptimizer : public Optimizer<T> {
public:
    NGPAdamOptimizer(const json& params) {
        update_hyperparams(params);
    }

    void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& grid_layer) override {
        m_n_weights = n_weights;
        m_n_hash_grid_weights = grid_layer[0].first;

        m_n_network_weights = grid_layer[1].first;

        m_first_moments.resize(m_n_weights);
        m_first_moments.memset(0);

        m_second_moments.resize(m_n_weights);
        m_second_moments.memset(0);

        m_param_steps.resize(m_n_weights);
        m_param_steps.memset(0);
    }

    void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
        ++m_current_step;

        uint32_t n_weights_to_optimize = n_weights();

        linear_kernel(ngp_adam_step<T>, 0, stream,
            n_weights_to_optimize,
            m_n_hash_grid_weights,
            m_n_hash_grid_weights + m_n_network_weights,
            loss_scale,
            m_base_learning_rate,
            m_beta1,
            m_beta2,
            m_epsilon,
            m_l2_reg,
            weights_full_precision,
            weights,
            gradients,
            m_first_moments.data(),
            m_second_moments.data(),
            m_param_steps.data()
        );
    }

    float learning_rate() const override {
        return m_base_learning_rate;
    }

    void set_learning_rate(float val) override {
        m_base_learning_rate = val;
    }

    uint32_t step() const override {
        return m_current_step;
    }

    uint32_t n_weights() const override {
        return m_n_weights;
    }

    T* custom_weights() const override {
        return nullptr;
    }

    uint32_t n_nested() const override {
        return 0;
    }

    void update_hyperparams(const json& params) override {
        if (params.contains("beta1")) {
            m_beta1 = params["beta1"];
        }

        if (params.contains("beta2")) {
            m_beta2 = params["beta2"];
        }

        if (params.contains("epsilon")) {
            m_epsilon = params["epsilon"];
        }

        if (params.contains("learning_rate")) {
            m_base_learning_rate = params["learning_rate"];
        }

        if (params.contains("l2_reg")) {
            m_l2_reg = params["l2_reg"];
        }

    }

    json hyperparams() const override {
        return {
            {"otype", "Adam"},
            {"beta1", m_beta1},
            {"beta2", m_beta2},
            {"epsilon", m_epsilon},
            {"learning_rate", m_base_learning_rate},
            {"l2_reg", m_l2_reg},
        };
    }

    json serialize() const override {
        json data;
        data["current_step"] = m_current_step;
        data["base_learning_rate"] = m_base_learning_rate;
        data["first_moments_binary"] = m_first_moments;
        data["second_moments_binary"] = m_second_moments;
        data["param_steps_binary"] = m_param_steps;
        return data;
    }

    void deserialize(const json& data) override {
        m_first_moments = data["first_moments_binary"];
        m_second_moments = data["second_moments_binary"];
        if (data.contains("param_steps_binary")) {
            m_param_steps = data["param_steps_binary"];
        } else {
            m_param_steps.resize(m_second_moments.size());
            m_param_steps.memset(0);
        }
        m_current_step = data["current_step"];
        m_base_learning_rate = data["base_learning_rate"];
    }

private:
    uint32_t m_n_weights;
    uint32_t m_n_hash_grid_weights;
    uint32_t m_n_network_weights;

    GPUMemory<float> m_first_moments;
    GPUMemory<float> m_second_moments;
    GPUMemory<uint32_t> m_param_steps;

    uint32_t m_current_step = 0;

    // Hyperparameters
    float m_base_learning_rate = 1e-3f;
    float m_beta1 = 0.9f;
    float m_beta2 = 0.999f;
    float m_epsilon = 1e-8f;
    float m_l2_reg = 1e-8f;

};

TCNN_NAMESPACE_END
