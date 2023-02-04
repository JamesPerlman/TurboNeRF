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

    NeRF(NerfNetwork network, OccupancyGrid occupancy_grid, BoundingBox bounding_box)
        : network(network)
        , occupancy_grid(occupancy_grid)
        , bounding_box(bounding_box)
    {};
    
    // TODO:
    // transform
    // bounding box (training, rendering)
    // masks
    // dist_paramss
};

NRC_NAMESPACE_END
