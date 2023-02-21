#pragma once

#include <cuda_runtime.h>
#include <future>
#include <vector>

#include "../common.h"
#include "../core/renderer.cuh"
#include "../models/render-pattern.cuh"
#include "../models/render-request.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFRenderingController {

private:

    uint32_t batch_size;

    std::vector<Renderer::Context> contexts;

    std::shared_ptr<RenderRequest> request = nullptr;

    Renderer renderer = Renderer();

public:
    NeRFRenderingController(const RenderPattern& pattern = RenderPattern::RectangularGrid, const uint32_t& batch_size = 0);

    void submit(
        std::shared_ptr<RenderRequest> request
    );

    void write_to(RenderTarget* target);

    void cancel();
};

NRC_NAMESPACE_END
