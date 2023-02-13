#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "../common.h"

/**
 * A workspace is a collection of allocations for GPU memory.
 * 
 */

NRC_NAMESPACE_BEGIN

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

    size_t total_size() const {
        return _total_size;
    }

    // free all allocations
    void free_allocations() {
        CUDA_CHECK_THROW(cudaSetDevice(device_id));
        for (const auto& allocation : _allocations) {
            if (allocation.ptr != nullptr) {
                CUDA_CHECK_THROW(cudaFreeAsync(allocation.ptr, 0));
            }
        }
        
        _allocations.clear();
    }

    ~Workspace() {
        try {
            free_allocations();
        } catch (const std::exception& e) {
            std::cout << "Error freeing workspace allocations: " << e.what() << std::endl;
        }
    }
};

NRC_NAMESPACE_END
