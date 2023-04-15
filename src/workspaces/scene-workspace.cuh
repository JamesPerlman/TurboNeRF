#pragma once

#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../core/occupancy-grid.cuh"
#include "../math/transform4f.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../utils/nerf-constants.cuh"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct SceneWorkspace: Workspace {

    using Workspace::Workspace;

    uint32_t n_nerfs = 0;

	Camera* camera;
    BoundingBox* bounding_boxes;
    OccupancyGrid* occupancy_grids;
    Transform4f* nerf_transforms;

    float* ray_t[2];
    float* ray_tmin[2];
    float* ray_tmax[2];
    bool* ray_active[2];
    uint32_t* intersect_nerfs;

    uint32_t n_rays;

    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_nerfs,
        const uint32_t& n_rays
    ) {
        free_allocations();
        camera          = allocate<Camera>(stream, 1);
        bounding_boxes  = allocate<BoundingBox>(stream, n_nerfs);
        occupancy_grids = allocate<OccupancyGrid>(stream, n_nerfs);
        nerf_transforms = allocate<Transform4f>(stream, n_nerfs);

        const uint32_t n_elems = n_rays * n_nerfs;

        ray_t[0]        = allocate<float>(stream, n_elems);
        ray_t[1]        = allocate<float>(stream, n_elems);

        ray_tmin[0]     = allocate<float>(stream, n_elems);
        ray_tmin[1]     = allocate<float>(stream, n_elems);

        ray_tmax[0]     = allocate<float>(stream, n_elems);
        ray_tmax[1]     = allocate<float>(stream, n_elems);

        ray_active[0]   = allocate<bool>(stream, n_elems);
        ray_active[1]   = allocate<bool>(stream, n_elems);

        intersect_nerfs = allocate<uint32_t>(stream, n_rays * NeRFConstants::n_max_nerfs / sizeof(uint32_t));

        this->n_nerfs = n_nerfs;
        this->n_rays = n_rays;
    }
};

TURBO_NAMESPACE_END
