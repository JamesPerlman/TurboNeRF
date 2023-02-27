#pragma once

#include <vector>

#include "../../models/render-pattern.cuh"
#include "../../models/render-request.cuh"
#include "../../models/render-task.cuh"

NRC_NAMESPACE_BEGIN

class RenderTaskFactory {
public:
    const int n_rays_per_task;
    const int n_samples_per_task;

    RenderTaskFactory(
        const int n_rays_per_task,
        const int n_samples_per_task
    ) : n_rays_per_task(n_rays_per_task), n_samples_per_task(n_samples_per_task) {};
    
    virtual std::vector<RenderTask> create_tasks(const RenderRequest* request) = 0;

    virtual bool can_preview() const {
        return false;
    }
};

RenderTaskFactory* create_render_task_factory(
    const RenderPattern& pattern,
    const int& n_rays_per_task,
    const int& n_samples_per_task
);

NRC_NAMESPACE_END
