#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <vector>

#include "../common.h"

TURBO_NAMESPACE_BEGIN

// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
template <typename T>
inline NRC_HOST_DEVICE T next_power_of_two(T v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
};

__global__ void get_1s_per_uint32(
    const size_t n_elements,
    const uint32_t* __restrict__ buffer,
    uint8_t* __restrict__ counts
);

template <typename T>
uint32_t count_1s(const T* buffer, uint8_t* counts, const size_t& n_elements, cudaStream_t stream) {
    // If the buffer's size is not an equal multiple of sizeof(uint32_t), we'll count the remaining bits on the CPU
    const size_t size_in_bytes = n_elements * sizeof(T);
    const size_t num_uint32s = size_in_bytes / sizeof(uint32_t);
    const int num_remaining_bytes = size_in_bytes % sizeof(uint32_t);

    // count 1s in each uint32
    const size_t num_threads = 256;
    const size_t num_blocks = (num_uint32s + num_threads - 1) / num_threads;

    get_1s_per_uint32<<<num_blocks, num_threads, 0, stream>>>(num_uint32s, reinterpret_cast<const uint32_t*>(buffer), counts);

    // use a parallel reduction to sum the counts
    uint32_t total_count = thrust::reduce(
        thrust::cuda::par.on(stream),
        counts,
        counts + num_uint32s,
        (uint32_t)0,
        thrust::plus<uint32_t>()
    );

    // count the remaining bits on the CPU if necessary
    if (num_remaining_bytes > 0) {
        std::vector<uint8_t> remaining_bytes(num_remaining_bytes);
        CUDA_CHECK_THROW(
            cudaMemcpy(
                remaining_bytes.data(),
                buffer + (size_in_bytes - num_remaining_bytes) * sizeof(uint8_t),
                num_remaining_bytes,
                cudaMemcpyDeviceToHost
            )
        );

        for (int i = 0; i < num_remaining_bytes; ++i) {
            for (int j = 0; j < 8; ++j) {
                total_count += (remaining_bytes[i] >> j) & 1;
            }
        }
    }

    return total_count;
}

TURBO_NAMESPACE_END
