#pragma once

#include <vector>

#include "../../models/render-pattern.cuh"
#include "../../models/render-request.cuh"
#include "../../models/render-task.cuh"

TURBO_NAMESPACE_BEGIN

class RenderTaskFactory {
public:
    const int n_rays_per_task;
    const int n_rays_per_preview;

    RenderTaskFactory(
        const int n_rays_per_task,
        const int n_rays_per_preview = 0
    ) : n_rays_per_task(n_rays_per_task), n_rays_per_preview(n_rays_per_preview) {};
    
    virtual std::vector<RenderTask> create_tasks(const RenderRequest* request) = 0;

    virtual bool can_preview() const {
        return false;
    }
};

RenderTaskFactory* create_render_task_factory(
    const RenderPattern& pattern,
    const int& n_rays_per_task,
    const int& n_rays_per_preview = 0
);

TURBO_NAMESPACE_END
