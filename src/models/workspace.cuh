#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct Workspace {
    struct Allocation {
        void* ptr;
        cudaStream_t stream;

        Allocation(const cudaStream_t& stream, void* ptr = nullptr) : ptr(ptr), stream(stream) {};
    };

private:
    std::vector<Allocation> _allocations;

public:
    // allocate memory and save it for freedom later
    template <typename T>
    T* allocate(const cudaStream_t& stream, const size_t& n_elements) {
        T* ptr;
        CUDA_CHECK_THROW(cudaMallocAsync(&ptr, n_elements * sizeof(T), stream));
        _allocations.emplace_back(stream, ptr);
        return ptr;
    }

    // free all allocations
    void free_allocations() {
        for (const auto& allocation : _allocations) {
            if (allocation.ptr != nullptr) {
                cudaFreeAsync(allocation.ptr, allocation.stream);
            }
        }

        _allocations.clear();
    }

    ~Workspace() {
        free_allocations();
    }
};

NRC_NAMESPACE_END
