#pragma once

#include <vector>
#include <memory>

#include "../rendering/ray-batch-coordinators/ray-batch-coordinator.cuh"
#include "../common.h"
#include "camera.cuh"
#include "nerf-renderable.cuh"
#include "render-modifiers.cuh"

TURBO_NAMESPACE_BEGIN

/**
 * A RenderTask is a portion of work that is performed by the Renderer.
 * It is a subset of the total work that must be done for a RenderRequest.
 */

struct RenderTask {
private:
    bool _canceled = false;

public:
    const int device_id;
    const int n_rays;
    const Camera camera;
    std::vector<NeRFRenderable> renderables;
    const RenderModifiers modifiers;

    // this is a bit smelly.  batch_coordinator performs the task of creating a ray batch and writing the results to the output buffer
    std::unique_ptr<RayBatchCoordinator> batch_coordinator;

    RenderTask(
        const int& device_id,
        const int& n_rays,
        const Camera& camera,
        const std::vector<NeRFRenderable>& renderables,
        const RenderModifiers& modifiers,
        std::unique_ptr<RayBatchCoordinator> batch_coordinator
    )
        : device_id(device_id)
        , n_rays(n_rays)
        , camera(camera)
        , renderables(renderables)
        , modifiers(modifiers)
        , batch_coordinator(std::move(batch_coordinator))
    { };

    RenderTask() : device_id(0), n_rays(0), camera(), renderables(), batch_coordinator(nullptr) { };

    void cancel() {
        _canceled = true;
    };

    bool is_canceled() const {
        return _canceled;
    };
};

TURBO_NAMESPACE_END
