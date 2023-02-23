#include "render-task-factory.cuh"
#include "hexagonal-grid-render-task-factory.cuh"
#include "rectangular-grid-render-task-factory.cuh"

NRC_NAMESPACE_BEGIN

RenderTaskFactory* create_render_task_factory(
    const RenderPattern& pattern,
    const int& n_rays_per_task,
    const int& n_samples_per_task
) {
    switch(pattern) {
        case RenderPattern::HexagonalGrid:
            return new HexagonalGridRenderTaskFactory(n_rays_per_task, n_samples_per_task);
        case RenderPattern::RectangularGrid:
            return new RectangularGridRenderTaskFactory(n_rays_per_task, n_samples_per_task);
    }
    return nullptr;
}

NRC_NAMESPACE_END
