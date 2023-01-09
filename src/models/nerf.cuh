#pragma once

#include "../common.h"

#include "cascaded-occupancy-grid.cuh"
#include "nerf-network.h"

NRC_NAMESPACE_BEGIN

struct NeRF {
    NerfNetwork network;
    CascadedOccupancyGrid occupancy_grid;

    NeRF(NerfNetwork network, CascadedOccupancyGrid occupancy_grid)
        : network(network), occupancy_grid(occupancy_grid) {};
    
    // TODO:
    // transform
    // bounding box (training, rendering)
    // masks
    // distortions
};

NRC_NAMESPACE_END
