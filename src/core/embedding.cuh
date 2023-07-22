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

/** @file   embedding.cuh
 *  @author James Perlman
 *  @brief  A general embedding layer implementation for tiny-cuda-nn.
 *          Much of this code has been adapted to conform to tcnn's style, thus the NVIDIA license.
 */

#pragma once

#include "../common.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>

TURBO_NAMESPACE_BEGIN

using namespace tcnn;

// these are hardcoded for row-major data

template <typename PARAMS_T>
__global__ void embedding_forward_kernel(
    const uint32_t n_elements,
    const uint32_t stride,
    const uint32_t n_dim,
    const PARAMS_T* __restrict__ params,
    const uint32_t* __restrict__ indices,
    PARAMS_T* __restrict__ output
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) {
        return;
    }

    const uint32_t vocab_idx = indices[idx];

    const PARAMS_T* embedding = params + vocab_idx * n_dim;
    PARAMS_T* out = output + idx;

    for (uint32_t i = 0; i < n_dim; ++i) {
        *out = embedding[i];
        out += stride;
    }
}

template <typename PARAMS_T>
__global__ void embedding_backward_kernel(
    const uint32_t n_vocab,
    const uint32_t n_dim,
    const uint32_t n_elements,
    const uint32_t stride,
    const uint32_t n_rays_per_image,
    const PARAMS_T* __restrict__ dL_doutput,
    const uint32_t* __restrict__ indices,
    PARAMS_T* __restrict__ param_gradients
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vocab * n_dim) {
        return;
    }

    const uint32_t vocab_idx = idx / n_dim;
    const uint32_t dim_idx = idx % n_dim;

    PARAMS_T grad = 0.0f;

    for (int i = 0; i < n_rays_per_image; ++i) {
        grad += dL_doutput[dim_idx * stride + vocab_idx * n_rays_per_image + i];
    }

    param_gradients[idx] = grad;
}

template <typename PARAMS_T>
__global__ void copy_embedding_gradients_kernel(
    const uint32_t n_params,
    const float scale,
    const float* __restrict__ grad_fp,
    PARAMS_T* __restrict__ grad_hp
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_params) {
        return;
    }

    grad_hp[idx] = (PARAMS_T)(*grad_fp / scale);
}

template <typename T, typename PARAMS_T=T, typename COMPUTE_T=T>
class Embedding {
public:
    Embedding(const uint32_t& n_vocab, const uint32_t& n_dim) : m_n_vocab(n_vocab), m_n_dim(n_dim) {};

	static uint32_t REQUIRED_ALIGNMENT() {
		return 16; // Just a hunch
	}

    uint32_t input_width() const {
        return m_n_dim;
    }

    uint32_t padded_output_width() const {
        return next_multiple(m_n_dim, REQUIRED_ALIGNMENT());
    }

    PARAMS_T* params() const {
        return m_params;
    }

    PARAMS_T* inference_params() const {
        return m_inference_params;
    }

    uint32_t n_params() const {
        return m_n_vocab * m_n_dim;
    }

    uint32_t n_embeddings() const {
        return m_n_vocab;
    }
    
    void set_params(PARAMS_T* params, PARAMS_T* inference_params, PARAMS_T* gradients) {
		m_params = params;
        m_inference_params = inference_params;
		m_gradients = gradients;
	}

    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1.0f) {
        generate_random_uniform<float>(rnd, n_params(), params_full_precision, 0.0f, scale);
    }

    void forward(
		cudaStream_t stream,
        const uint32_t& n_elements,
		const GPUMatrixDynamic<uint32_t>& input,
		GPUMatrixDynamic<COMPUTE_T>* output,
		bool use_inference_params = false
	) {
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(output->m() == padded_output_width());
		CHECK_THROW(input.n() == output->n());
        CHECK_THROW(input.n() % batch_size_granularity == 0);

        CHECK_THROW(this->n_params() > 0);
        
        if (use_inference_params) {
            CHECK_THROW(this->inference_params() != nullptr);
        } else {
            CHECK_THROW(this->params() != nullptr);
        }

        embedding_forward_kernel<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(
            n_elements,
            input.n(),
            m_n_dim,
            use_inference_params ? m_inference_params : m_params,
            input.data(),
            output->data()
        );
    }

    void backward(
		cudaStream_t stream,
        const uint32_t& n_elements,
        const uint32_t& n_rays_per_image,
		const GPUMatrixDynamic<uint32_t>& input,
		const GPUMatrixDynamic<COMPUTE_T>& dL_doutput,
		bool use_inference_params = false
	) {
		// Width
		CHECK_THROW(input.m() == input_width());
		CHECK_THROW(dL_doutput.m() == padded_output_width());

		// Batch size
		CHECK_THROW(input.n() == dL_doutput.n());
        CHECK_THROW(input.n() % batch_size_granularity == 0);

		// Param & gradient memory must have been set via `set_params(...)`
		CHECK_THROW(this->n_params() > 0);
        CHECK_THROW(this->params() != nullptr);

        embedding_backward_kernel<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(
            m_n_vocab,
            m_n_dim,
            n_elements,
            input_width(),
            n_rays_per_image,
            dL_doutput.data(),
            input.data(),
            m_params
        );
    }

    void inference_mixed_precision(
        cudaStream_t stream,
        const uint32_t& n_elements,
        const GPUMatrixDynamic<uint32_t>& indices,
        GPUMatrixDynamic<COMPUTE_T>& output
    ) {
        forward(stream, n_elements, indices, &output, true);
    }

private:
    uint32_t m_n_vocab = 0;
    uint32_t m_n_dim = 0;
    PARAMS_T* m_params;
    PARAMS_T* m_inference_params;
    PARAMS_T* m_gradients;
};

TURBO_NAMESPACE_END
