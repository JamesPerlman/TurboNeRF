#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "../common.h"

TURBO_NAMESPACE_BEGIN

struct RuntimeVersion {
    const int major;
    const int minor;
    const int subminor;

    static constexpr RuntimeVersion CompiledRuntimeVersion() {
        return RuntimeVersion{
            CUDART_VERSION / 1000,
            (CUDART_VERSION % 1000) / 10,
            CUDART_VERSION % 10
        };
    }

    std::string to_string() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(subminor);
    }

    bool operator>=(const RuntimeVersion& other) const {
        if (this->major > other.major) {
            return true;
        }
        if (this->major == other.major && this->minor > other.minor) {
            return true;
        }
        if (this->major == other.major && this->minor == other.minor && this->subminor >= other.subminor) {
            return true;
        }
        return false;
    }
};



struct DeviceArchitecture {
    const int major;
    const int minor;

    DeviceArchitecture(int major, int minor) : major(major), minor(minor) {};

    std::string to_string() const {
        return std::to_string(major) + "." + std::to_string(minor);
    }

    bool operator==(const DeviceArchitecture& other) const {
        return this->major == other.major && this->minor == other.minor;
    }

    bool operator!=(const DeviceArchitecture& other) const {
        return !(*this == other);
    }

    // https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    std::string arch_name() const {
        if (major == 6 && minor == 0) {
            return "Pascal";
        } else if (major == 6 && minor == 1) {
            return "Pascal";
        } else if (major == 6 && minor == 2) {
            return "Pascal";
        } else if (major == 7 && minor == 0) {
            return "Volta";
        } else if (major == 7 && minor == 2) {
            return "Volta";
        } else if (major == 7 && minor == 5) {
            return "Turing";
        } else if (major == 8 && minor == 0) {
            return "Ampere";
        } else if (major == 8 && minor == 6) {
            return "Ampere";
        } else if (major == 8 && minor == 7) {
            return "Ampere";
        } else if (major == 8 && minor == 9) {
            return "Lovelace";
        } else if (major == 9 && minor == 0) {
            return "Hopper";
        } else {
            return "Unknown";
        }
    }
};

struct RuntimeManager {
    private:
    
    // CMake adds a define called CUDA_ARCHS, which is a string of the form "61,75,80"
    std::vector<DeviceArchitecture> parse_cuda_archs() {
        const std::string arch_string = std::string(CUDA_ARCHS);
        std::vector<DeviceArchitecture> architectures;
        std::stringstream ss(arch_string);
        int arch;

        while (ss >> arch) {
            const int major = arch / 10;
            const int minor = arch % 10;

            architectures.emplace_back(major, minor);
            
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        return architectures;
    }

    public:

    const std::vector<DeviceArchitecture> cuda_archs;

    RuntimeManager() : cuda_archs(parse_cuda_archs()) {};
        
    static RuntimeVersion current_runtime_version() {
        int driver_version = 0;
        CUDA_CHECK_THROW(cudaDriverGetVersion(&driver_version));

        return RuntimeVersion{
            driver_version / 1000,
            (driver_version % 1000) / 10,
            driver_version % 10
        };
    }

    static DeviceArchitecture get_device_architecture(int device_id) {
        cudaDeviceProp prop;
        CUDA_CHECK_THROW(cudaGetDeviceProperties(&prop, device_id));
        return {prop.major, prop.minor};
    }

    // https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility
    static constexpr RuntimeVersion required_runtime_version() {
       return RuntimeVersion::CompiledRuntimeVersion();
    }

    static bool is_driver_version_sufficient() {
        return current_runtime_version() >= required_runtime_version();
    }

    static bool is_architecture_supported(const std::vector<DeviceArchitecture>& supported_archs, const DeviceArchitecture& device_arch) {
        for (const DeviceArchitecture& arch : supported_archs) {
            if (device_arch == arch) {
                return true;
            }
        }
        return false;
    }

    bool check_runtime() {
        printf("Checking CUDA runtime and device capabilities...\n");
        if (!is_driver_version_sufficient()) {
            printf(
                "CUDA runtime version is %s which is not sufficient.  This binary was compiled for CUDA runtime version %s.\n",
                current_runtime_version().to_string().c_str(),
                required_runtime_version().to_string().c_str()
            );
            
            printf("Please upgrade your GPU driver to the latest version: https://www.nvidia.com/download/index.aspx\n");

            return false;
        }

        for (int device_id = 0; device_id < cuda_archs.size(); device_id++) {
            DeviceArchitecture device_arch = get_device_architecture(device_id);
            if (!is_architecture_supported(cuda_archs, device_arch)) {
                printf("Device %d with compute capability %s (%s) is not supported.\n", device_id, device_arch.to_string().c_str(), device_arch.arch_name().c_str());
                printf("Supported architecture(s) are: \n");
                for (const DeviceArchitecture& arch : cuda_archs) {
                    printf("%s (%s)\n", arch.to_string().c_str(), arch.arch_name().c_str());
                }
                printf("Please double-check your installation and GPU.\n");
                return false;
            }

            printf(
                "Device %d with compute capability %s (%s) is supported!\n",
                device_id,
                device_arch.to_string().c_str(),
                device_arch.arch_name().c_str()
            );
        }

        printf(
            "CUDA runtime is %s which is sufficient to run this binary which was built for %s. We're good to go!\n",
            current_runtime_version().to_string().c_str(),
            required_runtime_version().to_string().c_str()
        );

        return true;
    }

};

TURBO_NAMESPACE_END
