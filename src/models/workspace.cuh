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
    T* allocate(const size_t& n_bytes, const cudaStream_t& stream) {
        T* ptr;
        CUDA_CHECK_THROW(cudaMallocAsync(&ptr, n_bytes, stream));
        _allocations.emplace_back(ptr);
        return ptr;
    }

    // This function returns a lambda that can be used to allocate memory on the device.
    // auto make_allocator(const cudaStream_t& stream) const {
    //     return [this, stream]<typename T>(const size_t& n_bytes) {
    //         return allocate<T>(n_bytes, stream);
    //     };
    // }

    // free all allocations
    ~Workspace() {
        for (int i = 0; i < _allocations.size(); ++i) {
            if (_allocations[i] != nullptr) {
                cudaFreeAsync(_allocations[i], _streams[i]);
            }
        }
    }
};

NRC_NAMESPACE_END
