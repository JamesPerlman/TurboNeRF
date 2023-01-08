#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "../common.h"

NRC_NAMESPACE_BEGIN

struct Workspace {
private:
    std::vector<void*> _allocations;
    std::vector<cudaStream_t> _streams;

public:
    // allocate memory and save it for freedom later
    template <typename T>
    T* allocate(const cudaStream_t& stream, const size_t& n_elements) {
        T* ptr;
        CUDA_CHECK_THROW(cudaMallocAsync(&ptr, n_elements * sizeof(T), stream));
        _allocations.emplace_back(ptr);
        return ptr;
    }

    // free all allocations
    void free_allocations() {
        for (int i = 0; i < _allocations.size(); ++i) {
            if (_allocations[i] != nullptr) {
                cudaFreeAsync(_allocations[i], _streams[i]);
            }
        }
        
        _allocations.clear();
    }

    ~Workspace() {
        free_allocations();
    }
};

NRC_NAMESPACE_END
