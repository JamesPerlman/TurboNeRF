#pragma once

#include "../common.h"

#include "network-params-workspace.cuh"
#include "network-workspace.cuh"
#include "occupancy-grid-workspace.cuh"
#include "rendering-workspace.cuh"
#include "training-workspace.cuh"
#include "workspace-ref.cuh"

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

struct WorkspaceRef {
    const NetworkParamsWorkspace* network_params_ws = nullptr;
    const NetworkWorkspace* network_ws = nullptr;
    const OccupancyGridWorkspace* occupancy_grid_ws = nullptr;
    const RenderingWorkspace* rendering_ws = nullptr;
    const TrainingWorkspace* training_ws = nullptr;

    WorkspaceRef(WorkspaceType type) {
        switch (type) {
            case NETWORK_PARAMS:
                network_params_ws = new NetworkParamsWorkspace();
                break;
            case NETWORK:
                network_ws = new NetworkWorkspace();
                break;
            case OCCUPANCY_GRID:
                occupancy_grid_ws = new OccupancyGridWorkspace();
                break;
            case RENDERING:
                rendering_ws = new RenderingWorkspace();
                break;
            case TRAINING:
                training_ws = new TrainingWorkspace();
                break;
        }
    }
};

NRC_NAMESPACE_END
