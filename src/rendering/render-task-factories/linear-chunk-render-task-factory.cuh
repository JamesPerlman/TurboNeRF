#pragma once

#include "../../math/hexagon-grid.cuh"
#include "../ray-batch-coordinators/linear-buffer-ray-batch-coordinator.cuh"
#include "render-task-factory.cuh"

TURBO_NAMESPACE_BEGIN

class LinearChunkRenderTaskFactory : public RenderTaskFactory {
public:
    using RenderTaskFactory::RenderTaskFactory;

    bool can_preview() const override {
        return false;
    }

    // TODO: this needs some revision/optimization.  It could be better at optimizing hexagon tiling.
    std::vector<RenderTask> create_tasks(const RenderRequest* request) override {
        
        const int n_pixels_total = request->output->n_pixels();

        const int n_tasks = tcnn::div_round_up(n_pixels_total, n_rays_per_task);

        // create tasks
        std::vector<RenderTask> tasks;
        tasks.reserve(n_tasks);

        // next we create a task for each hexagonal tile to fill it in with higher detail.
        for (int i = 0; i < n_tasks; ++i) {
            const int start_idx = i * n_rays_per_task;
            const int n_rays = std::min(n_rays_per_task, n_pixels_total - start_idx);
            tasks.emplace_back(
                n_rays,
                request->camera,
                request->proxies[0]->get_nerf_ptrs(),
                std::unique_ptr<RayBatchCoordinator>(
                    new LinearBufferRayBatchCoordinator(
                        n_rays,
                        start_idx
                    )
                )
            );
        }

        return tasks;
    }
};

TURBO_NAMESPACE_END
