#pragma once

#include <cuda_runtime.h>
#include <future>
#include <vector>

#include "../common.h"
#include "../core/renderer.cuh"
#include "../models/render-pattern.cuh"
#include "../models/render-request.cuh"

TURBO_NAMESPACE_BEGIN

struct NeRFRenderingController {

private:

    uint32_t batch_size;

    std::vector<Renderer::Context> contexts;

    std::shared_ptr<RenderRequest> request = nullptr;
    std::vector<RenderTask> tasks;

    Renderer renderer = Renderer();

    RenderPattern pattern;

    std::vector<size_t> get_cuda_memory_allocated() const;

public:
    NeRFRenderingController(const RenderPattern& pattern = RenderPattern::LinearChunks, const uint32_t& batch_size = 0);

    float min_step_size = NeRFConstants::min_step_size;

    void submit(
        std::shared_ptr<RenderRequest> request
    );

    void cancel();
};

TURBO_NAMESPACE_END
