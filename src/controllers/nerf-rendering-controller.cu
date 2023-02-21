#include <chrono>

#include "nerf-rendering-controller.h"
#include "../services/device-manager.cuh"

using namespace nrc;
using namespace tcnn;

void render_thread_fn(NeRFRenderingController* controller, RenderRequest& request);

NeRFRenderingController::NeRFRenderingController(
    const RenderPattern& pattern,
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

void NeRFRenderingController::cancel() {
    if (request != nullptr)
        request->cancel();
}

void NeRFRenderingController::submit(
    std::shared_ptr<RenderRequest> request
) {
    // TODO: batching/chunking/distributing requests across multiple GPUs
    auto& ctx = contexts[0];
    
    this->request = request;

    renderer.submit(ctx, request.get());
}

void NeRFRenderingController::write_to(
    RenderTarget* target
) {
    auto& ctx = contexts[0];
    renderer.write_to(ctx, target);
}
