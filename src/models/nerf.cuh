#pragma once

#include "../common.h"

#include "bounding-box.cuh"
#include "../core/occupancy-grid.cuh"
#include "../core/nerf-network.cuh"

NRC_NAMESPACE_BEGIN

struct NeRF {
    NerfNetwork network;
    OccupancyGrid occupancy_grid;
    BoundingBox bounding_box;

    const int device_id;

    NeRF(const int& device_id, BoundingBox bounding_box)
        : device_id(device_id)
        , bounding_box(bounding_box)
        , network(device_id, bounding_box.size_x)
        , occupancy_grid(device_id, OccupancyGrid::get_max_n_levels(bounding_box.size_x), 128)
    { };
    
    // TODO:
    // transform
    // bounding box (training, rendering)
    // masks
    // distortions :D
};

NRC_NAMESPACE_END
