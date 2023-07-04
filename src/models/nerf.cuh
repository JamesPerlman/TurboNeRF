#pragma once

#include <memory>

#include "../common.h"

#include "../core/occupancy-grid.cuh"
#include "../core/nerf-network.cuh"
#include "../workspaces/dataset-workspace.cuh"
#include "bounding-box.cuh"

TURBO_NAMESPACE_BEGIN

// forward declare NeRFProxy
struct NeRFProxy;

struct NeRF {
    DatasetWorkspace dataset_ws;
    NetworkParamsWorkspace params;
    OccupancyGrid occupancy_grid;
    NerfNetwork network;
    NeRFProxy* proxy;

    const int device_id;
    
    bool is_image_data_loaded = false;

    NeRF(const int& device_id, NeRFProxy* proxy);

    void free_device_memory();
    
};

TURBO_NAMESPACE_END
