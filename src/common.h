#pragma once

#include <tiny-cuda-nn/common.h>

#define TURBO_NAMESPACE_BEGIN namespace turbo {
#define TURBO_NAMESPACE_END }

#define NRC_HOST_DEVICE __host__ __device__

#define CHECK_DATA(varname, data_type, data_ptr, data_size, stream) \
    std::vector<data_type> varname(data_size); \
    CUDA_CHECK_THROW(cudaMemcpyAsync(varname.data(), data_ptr, data_size * sizeof(data_type), cudaMemcpyDeviceToHost, stream)); \
    cudaStreamSynchronize(stream);

#define READWRITE_PROPERTY(type, name, default) \
    private: \
        type name = default; \
    public: \
        type get_##name() const { return name; } \
        void set_##name(type value) { name = value; }

#define READONLY_PROPERTY(type, name, default) \
    private: \
        type name = default; \
    public: \
        type get_##name() const { return name; }

#define CURAND_ASSERT_SUCCESS(curand_call) \
    do { \
        curandStatus_t curand_status = curand_call; \
        if (curand_status != CURAND_STATUS_SUCCESS) { \
            std::stringstream ss; \
            ss << "CURAND error: " << curand_status; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0);
