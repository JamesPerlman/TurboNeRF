#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "../models/nerf-proxy.cuh"
#include "../utils/parallel-utils.cuh"
#include "../common.h"

TURBO_NAMESPACE_BEGIN

#define TURBO_NERF_FILE_VERSION 0

struct FileManager {
    struct NeRFFileData {
        char project_name[16];
        char dataset_path[256];
        uint32_t version;
        uint32_t n_color_params;
        uint32_t n_density_params;
        uint32_t n_occ_grid_bits;
        uint32_t aabb_scale;
    };
    
    static void save(
        const NeRFProxy* proxy,
        const std::filesystem::path& file_path,
        const cudaStream_t& stream = 0
    ) {
        std::ofstream file(file_path, std::ios::binary);

        if (!file) {
            throw std::runtime_error("Could not open file for writing");
        }

        const NeRF& nerf = proxy->nerfs[0];

        // prepare data
        NeRFFileData data;
        strncpy(data.project_name, "TurboNeRF", sizeof(data.project_name));
        
        data.version = TURBO_NERF_FILE_VERSION;

        std::string dataset_path = proxy->dataset->file_path->string();
        size_t copy_len = std::min(dataset_path.size(), sizeof(data.dataset_path) - 1);
        strncpy(data.dataset_path, dataset_path.c_str(), copy_len);
        data.dataset_path[copy_len] = '\0';

        data.n_color_params = nerf.params.n_color_params;
        data.n_density_params = nerf.params.n_density_params;
        data.n_occ_grid_bits = nerf.occupancy_grid.workspace.n_bitfield_elements;
        data.aabb_scale = static_cast<uint32_t>(proxy->bounding_box.size());

        // copy nerf params data to CPU
        size_t n_params = data.n_color_params + data.n_density_params;
        std::vector<float> params(n_params);
        CUDA_CHECK_THROW(
            cudaMemcpy(
                params.data(),
                proxy->nerfs[0].params.params_fp,
                n_params * sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );

        // copy occupancy grid bitfield to CPU
        std::vector<uint8_t> occ_grid_bits(data.n_occ_grid_bits);

        CUDA_CHECK_THROW(
            cudaMemcpy(
                occ_grid_bits.data(),
                proxy->nerfs[0].occupancy_grid.get_bitfield(),
                data.n_occ_grid_bits * sizeof(uint8_t),
                cudaMemcpyDeviceToHost
            )
        );

        // write data to file
        file.write(reinterpret_cast<char*>(&data), sizeof(data));

        std::streamsize param_bytes = n_params * sizeof(float) / sizeof(char);
        file.write(reinterpret_cast<char*>(params.data()), param_bytes);

        std::streamsize occ_grid_bytes = data.n_occ_grid_bits * sizeof(uint8_t) / sizeof(char);
        file.write(reinterpret_cast<char*>(occ_grid_bits.data()), occ_grid_bytes);
    }

    static void load(
        NeRFProxy* proxy,
        const std::filesystem::path& file_path,
        const cudaStream_t& stream = 0
    ) {
        std::ifstream file(file_path, std::ios::binary);

        if (!file) {
            throw std::runtime_error("Could not open file for reading");
        }

        // read data
        NeRFFileData data;
        file.read(reinterpret_cast<char*>(&data), sizeof(data));

        if (strncmp(data.project_name, "TurboNeRF", sizeof(data.project_name)) != 0) {
            throw std::runtime_error("File is not a TurboNeRF file");
        }

        if (data.version != TURBO_NERF_FILE_VERSION) {
            throw std::runtime_error(
                fmt::format("File version {} is not supported. Supported version is {}", data.version, TURBO_NERF_FILE_VERSION)
            );
        }

        // load dataset
        std::string dataset_path_str(data.dataset_path);
        // TODO: fix this. it doesn't really do anything, the dataset is unusable here
        proxy->dataset = Dataset(dataset_path_str);
        
        // load nerf params
        size_t n_params = data.n_color_params + data.n_density_params;
        std::vector<float> params(n_params);
        std::streamsize param_bytes = n_params * sizeof(float) / sizeof(char);
        file.read(reinterpret_cast<char*>(params.data()), param_bytes);

        // need AABB to create the NeRFs, but this code should probably go somewhere else
        const BoundingBox bbox(static_cast<float>(data.aabb_scale));
        
        proxy->bounding_box = bbox;

        proxy->nerfs.reserve(DeviceManager::get_device_count());

        DeviceManager::foreach_device([&](const int device_id, const cudaStream_t& stream) {
            proxy->nerfs.emplace_back(device_id, proxy);
        });

        // prep NeRF params
        auto& nerf = proxy->nerfs[0];
        nerf.params.enlarge(stream, data.n_density_params, data.n_color_params);
        
        // copy to GPU
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                nerf.params.params_fp,
                params.data(),
                param_bytes,
                cudaMemcpyHostToDevice,
                stream
            )
        );

        // need to copy half-precision params too
		copy_and_cast<tcnn::network_precision_t, float>(
			stream,
			n_params,
			nerf.params.params_hp,
			nerf.params.params_fp
		);

        // load occupancy grid bitfield
        std::vector<uint8_t> occ_grid_bits(data.n_occ_grid_bits);
        std::streamsize occ_grid_bytes = data.n_occ_grid_bits * sizeof(uint8_t) / sizeof(char);
        file.read(reinterpret_cast<char*>(occ_grid_bits.data()), occ_grid_bytes);

        // prep occupancy grid
        auto& occ_grid = nerf.occupancy_grid;
        occ_grid.initialize(stream, false);
        
        // copy to GPU
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                occ_grid.get_bitfield(),
                occ_grid_bits.data(),
                occ_grid_bytes,
                cudaMemcpyHostToDevice,
                stream
            )
        );

        // synchronize stream
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		proxy->is_valid = true;
	    proxy->can_render = true;
		proxy->can_train = false;
    }
};

TURBO_NAMESPACE_END
