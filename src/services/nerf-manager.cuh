#pragma once

#include <memory>
#include <stdint.h>
#include <vector>

#include "../common.h"
#include "../models/cascaded-occupancy-grid.cuh"
#include "../models/dataset.h"
#include "../models/nerf-network.h"
#include "../models/nerf.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFManager {
private:
    std::vector<NeRF> nerfs;

public:
    // get nerfs
    const std::vector<NeRF>& get_nerfs() const {
        return nerfs;
    }

    // create a new nerf
    NeRF& create_trainable_nerf(const cudaStream_t& stream, const float& aabb_size) {
        NerfNetwork network(aabb_size);
        CascadedOccupancyGrid occupancy_grid(5);

        // for now we will initialize the occupancy grid here, but it should probably done somewhere else
        occupancy_grid.initialize_bitfield(stream);
        occupancy_grid.initialize_values(stream);
        
        // Initialize occupancy grid bitfield (all bits set to 1)
        CUDA_CHECK_THROW(
            cudaMemsetAsync(
                occupancy_grid.get_bitfield(),
                (uint8_t)0b11111111, // set all bits to 1
                occupancy_grid.get_n_bitfield_elements(),
                stream
            )
        );

        nerfs.emplace_back(network, occupancy_grid);
        return nerfs.back();
    }

    // manage nerfs

    // destroy nerfs    
};

NRC_NAMESPACE_END
