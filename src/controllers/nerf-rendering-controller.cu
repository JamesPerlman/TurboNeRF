#include <chrono>

#include "nerf-rendering-controller.h"
#include "../services/device-manager.cuh"

using namespace nrc;
using namespace tcnn;

void render_thread_fn(NeRFRenderingController* controller, RenderRequest& request);

NeRFRenderingController::NeRFRenderingController(
    const uint32_t& batch_size
) {
    if (batch_size == 0) {
        // TODO: determine batch size from GPU specs
        this->batch_size = 1<<20;
    } else {
        this->batch_size = batch_size;
    }

    for (int i = 0; i < DeviceManager::get_device_count(); ++i) {
        contexts.emplace_back(
            DeviceManager::get_stream(i),
            RenderingWorkspace(i),
            this->batch_size
        );
    }
}

void NeRFRenderingController::submit(
    RenderRequest& request,
    bool async
) {
    // if we are still rendering in the background...
    if (render_future.valid() && render_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        // then just ignore this request
        return;
    } 

    // otherwise, we can start rendering
    
    // TODO: batching/chunking/distributing requests across multiple GPUs
    auto& ctx = contexts[0];

    render_future = std::async(
        std::launch::async,
        [this, &ctx, &request]() {
            renderer.submit(ctx, request);
        }
    );

    if (!async) {
        render_future.wait();
    }
}

void NeRFRenderingController::write_to(
    RenderTarget* target
) {
    auto& ctx = contexts[0];
    renderer.write_to(ctx, target);
}