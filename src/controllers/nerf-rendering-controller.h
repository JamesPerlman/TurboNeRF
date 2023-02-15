#pragma once

#include <cuda_runtime.h>
#include <future>
#include <vector>

#include "../common.h"
#include "../core/renderer.cuh"
#include "../models/render-request.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFRenderingController {

    NeRFRenderingController(
        const uint32_t& batch_size = 0
    );

    void submit(
        RenderRequest& request,
        bool async = false
    );

    void write_to(
        RenderTarget* target
    );

    bool is_rendering() const;

    void wait_until_finished() const;

private:

    uint32_t batch_size;

    std::vector<Renderer::Context> contexts;

    Renderer renderer = Renderer();

    std::future<void> render_future;
};

NRC_NAMESPACE_END
