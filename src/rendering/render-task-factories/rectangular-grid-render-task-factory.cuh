#pragma once

#include "../../utils/hexagon-grid.cuh"
#include "render-task-factory.cuh"

NRC_NAMESPACE_BEGIN

class RectangularGridRenderTaskFactory : public RenderTaskFactory {
public:
    using RenderTaskFactory::RenderTaskFactory;

    std::vector<RenderTask> create_tasks(const RenderRequest* request) override {
        // stub
        throw std::runtime_error("TODO: Implement RectangularGridRenderTaskFactory::create_tasks");
    }
};

NRC_NAMESPACE_END
