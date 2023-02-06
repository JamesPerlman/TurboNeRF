#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "../common.h"
#include "../core/renderer.cuh"
#include "../models/render-buffer.cuh"
#include "../models/render-request.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFRenderingController {

    NeRFRenderingController(
        const uint32_t& batch_size = 0
    );

    void request_render(
        RenderRequest& request
    );

private:

    uint32_t batch_size;

    uint32_t render_area;

    std::vector<Renderer::Context> contexts;

    Renderer renderer = Renderer();
};

NRC_NAMESPACE_END
