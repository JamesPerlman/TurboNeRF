#pragma once

#include "../common.h"

#include "../core/occupancy-grid.cuh"
#include "../core/nerf-network.cuh"
#include "../workspaces/dataset-workspace.cuh"
#include "bounding-box.cuh"

TURBO_NAMESPACE_BEGIN

struct NeRF {
    DatasetWorkspace dataset_ws;
    NetworkParamsWorkspace params;
    OccupancyGrid occupancy_grid;
    BoundingBox bounding_box;

    const int device_id;

    NeRF(const int& device_id, BoundingBox bounding_box)
        : device_id(device_id)
        , bounding_box(bounding_box)
        , dataset_ws(device_id)
        , params(device_id)
        , occupancy_grid(device_id, OccupancyGrid::get_max_n_levels(bounding_box.size_x), 128)
    { };
    
};

TURBO_NAMESPACE_END
