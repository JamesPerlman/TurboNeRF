#include "nerf-rendering-controller.h"
#include "../services/device-manager.cuh"

using namespace nrc;
using namespace tcnn;

NeRFRenderingController::NeRFRenderingController(
    uint32_t batch_size
) {
    if (batch_size == 0) {
        // TODO: determine batch size from GPU specs
        this->batch_size = 1<<21;
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

void NeRFRenderingController::request_render(
    RenderRequest& request
) {
    // TODO: batching/chunking/distributing requests across multiple GPUs
    auto& ctx = contexts[0];
    request.output.clear(ctx.stream);
    renderer.render(ctx, request);
}
