#pragma once

#include "../common.h"

#include "bounding-box.cuh"
#include "cascaded-occupancy-grid.cuh"
#include "nerf-network.h"

NRC_NAMESPACE_BEGIN

struct NeRF {
    NerfNetwork network;
    CascadedOccupancyGrid occupancy_grid;
    BoundingBox bounding_box;

    NeRF(NerfNetwork network, CascadedOccupancyGrid occupancy_grid, BoundingBox bounding_box)
        : network(network)
        , occupancy_grid(occupancy_grid)
        , bounding_box(bounding_box)
    {};
    
    // TODO:
    // transform
    // bounding box (training, rendering)
    // masks
    // distortions
};

NRC_NAMESPACE_END
