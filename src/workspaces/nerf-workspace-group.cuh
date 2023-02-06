#pragma once

#include "../common.h"

#include "network-params-workspace.cuh"
#include "network-workspace.cuh"
#include "occupancy-grid-workspace.cuh"
#include "rendering-workspace.cuh"
#include "training-workspace.cuh"

NRC_NAMESPACE_BEGIN

/**
 * A workspace ref holds a reference to a single workspace.
 * It is used by the workspace manager to keep track of the workspaces.
 */

enum WorkspaceType {
    NETWORK_PARAMS,
    NETWORK,
    OCCUPANCY_GRID,
    RENDERING,
    TRAINING,
};

struct NeRFWorkspaceGroup {
    NetworkParamsWorkspace params_ws;
    NetworkWorkspace network_ws;
    OccupancyGridWorkspace grid_ws;

    NeRFWorkspaceGroup(const int device_id)
        : params_ws(device_id)
        , network_ws(device_id)
        , grid_ws(device_id)
    { }
};

NRC_NAMESPACE_END
