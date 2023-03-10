#include "render-task-factory.cuh"
#include "linear-chunk-render-task-factory.cuh"
#include "hexagonal-grid-render-task-factory.cuh"
#include "rectangular-grid-render-task-factory.cuh"

TURBO_NAMESPACE_BEGIN

RenderTaskFactory* create_render_task_factory(
    const RenderPattern& pattern,
    const int& n_rays_per_task,
    const int& n_rays_per_preview
) {
    switch(pattern) {
        case RenderPattern::LinearChunks:
            return new LinearChunkRenderTaskFactory(n_rays_per_task, n_rays_per_preview);
        case RenderPattern::HexagonalGrid:
            return new HexagonalGridRenderTaskFactory(n_rays_per_task, n_rays_per_preview);
        case RenderPattern::RectangularGrid:
            return new RectangularGridRenderTaskFactory(n_rays_per_task, n_rays_per_preview);
    }
    return nullptr;
}

TURBO_NAMESPACE_END
