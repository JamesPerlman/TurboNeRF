#pragma once

#include "../rendering/ray-batch-coordinators/ray-batch-coordinator.cuh"
#include "../common.h"
#include "camera.cuh"
#include "nerf.cuh"

NRC_NAMESPACE_BEGIN

/**
 * A RenderTask is a portion of work that is performed by the Renderer.
 * It is a subset of the total work that must be done for a RenderRequest.
 */

struct RenderTask {
private:
    bool _canceled = false;

public:
    const int n_rays;

    const Camera camera;
    
    std::vector<NeRF*> nerfs;

    // this is a bit smelly.  batch_coordinator performs the task of creating a ray batch and writing the results to the output buffer
    std::unique_ptr<RayBatchCoordinator> batch_coordinator;

    RenderTask(
        const int& n_rays,
        const Camera& camera,
        const std::vector<NeRF*>& nerfs,
        std::unique_ptr<RayBatchCoordinator> batch_coordinator
    )
        : n_rays(n_rays)
        , camera(camera)
        , nerfs(nerfs)
        , batch_coordinator(std::move(batch_coordinator))
    { };

    RenderTask() : n_rays(0), camera(), nerfs(), batch_coordinator(nullptr) { };

    void cancel() {
        _canceled = true;
    };

    bool is_canceled() const {
        return _canceled;
    };
};

NRC_NAMESPACE_END
