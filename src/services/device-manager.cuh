#pragma once

#include <vector>

#include "../common.h"

NRC_NAMESPACE_BEGIN

class DeviceManager {
private:

    static DeviceManager& _get_instance() {
        static DeviceManager _instance;
        return _instance;
    }

    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    DeviceManager() {}

    int _device_count = 0;
    bool _has_queried_devices = false;
    std::vector<cudaStream_t> _streams;

    int _get_device_count() {
        if (!_has_queried_devices) {
            CUDA_CHECK_THROW(cudaGetDeviceCount(&_device_count));
        }
        // return _device_count;
        return 1;
    }

    const cudaStream_t& _get_stream(int device) {
        const int n_devices = _get_device_count();

        if (_streams.size() <= n_devices) {
            // if this is the first call, we need to create streams for each device
            
            int prev_device;
            CUDA_CHECK_THROW(cudaGetDevice(&prev_device));

            _streams.reserve(n_devices);

            for (int i = 0; i < n_devices; i++) {
                CUDA_CHECK_THROW(cudaSetDevice(i));
                cudaStream_t stream;
                CUDA_CHECK_THROW(cudaStreamCreate(&stream));
                _streams.emplace_back(stream);
            }

            CUDA_CHECK_THROW(cudaSetDevice(prev_device));
        }

        if (device >= n_devices) {
            throw std::runtime_error("Invalid device index");
        }

        return _streams[device];
    }

    ~DeviceManager() {
        for (int i = 0; i < _streams.size(); i++) {
            try {
                CUDA_CHECK_THROW(cudaSetDevice(i));
                CUDA_CHECK_THROW(cudaStreamDestroy(_streams[i]));
            } catch (const std::runtime_error& e) {
                std::cerr << "Error destroying stream: " << e.what() << std::endl;
            }
        }
        _streams.clear();
    }

public:

    static int get_device_count() {
        return _get_instance()._get_device_count();
    }

    static const cudaStream_t& get_stream(int device) {
        return _get_instance()._get_stream(device);
    }

    static void foreach_device(std::function<void(const int& device_id, const cudaStream_t& stream)> func) {
        int prev_device;
        CUDA_CHECK_THROW(cudaGetDevice(&prev_device));
        
        const int n_devices = get_device_count();

        for (int i = 0; i < n_devices; i++) {
            CUDA_CHECK_THROW(cudaSetDevice(i));
            func(i, _get_instance()._get_stream(i));
        }

        CUDA_CHECK_THROW(cudaSetDevice(prev_device));
    }

    static void synchronize() {
        foreach_device([](const int& device_id, const cudaStream_t& stream) {
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        });
    }
};

NRC_NAMESPACE_END
