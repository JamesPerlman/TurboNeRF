#pragma once

#include "../../math/hexagon-grid.cuh"
#include "render-task-factory.cuh"

TURBO_NAMESPACE_BEGIN

class RectangularGridRenderTaskFactory : public RenderTaskFactory {
public:
    using RenderTaskFactory::RenderTaskFactory;

    std::vector<RenderTask> create_tasks(const RenderRequest* request) override {
        // stub
        throw std::runtime_error("TODO: Implement RectangularGridRenderTaskFactory::create_tasks");
    }
};

TURBO_NAMESPACE_END
