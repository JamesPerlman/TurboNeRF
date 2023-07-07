#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "../common.h"

/**
 * A workspace is a collection of allocations for GPU memory.
 * 
 */

TURBO_NAMESPACE_BEGIN

struct Workspace {
    struct Allocation {
        void* ptr;

        Allocation(void* ptr = nullptr) : ptr(ptr) {};
    };

private:
    std::vector<Allocation> _allocations;
    size_t _total_size = 0;
public:
    const int device_id;

    Workspace(const int& device_id = 0) : device_id(device_id) { };
    
    // allocate memory and save it for freedom later
    template <typename T>
    T* allocate(const cudaStream_t& stream, const size_t& n_elements) {
        T* ptr;
        const size_t n_bytes = n_elements * sizeof(T);
        CUDA_CHECK_THROW(cudaMallocAsync(&ptr, n_bytes, stream));
        _total_size += n_bytes;
        _allocations.emplace_back(ptr);
        return ptr;
    }

    size_t get_bytes_allocated() const {
        return _total_size;
    }

    // free all allocations
    virtual void free_allocations() {
        CUDA_CHECK_THROW(cudaSetDevice(device_id));
        // this is not super robust, and does not cover the edge case where one allocation fails
        // in this case, even the freed allocations will stay in the _allocations vector
        for (auto& allocation : _allocations) {
            if (allocation.ptr != nullptr) {
                CUDA_CHECK_THROW(cudaFreeAsync(allocation.ptr, 0));
                allocation.ptr = nullptr;
            }
        }

        _allocations.clear();
        _total_size = 0;
    }

    // watchout - we are not freeing allocations in the destructor since workspaces are often shared.
    // this is truly horrible, but we need to manage memory manually
};

TURBO_NAMESPACE_END
