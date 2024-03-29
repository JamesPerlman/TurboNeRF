#include <chrono>
#include <memory>
#include <vector>

#include "nerf-rendering-controller.h"
#include "../rendering/render-task-factories/render-task-factory.cuh"
#include "../services/device-manager.cuh"

using namespace std;
using namespace tcnn;

TURBO_NAMESPACE_BEGIN

void render_thread_fn(NeRFRenderingController* controller, RenderRequest& request);

NeRFRenderingController::NeRFRenderingController(
    const RenderPattern& pattern,
    const uint32_t& batch_size
) : pattern(pattern) {
    if (batch_size == 0) {
        // TODO: determine batch size from GPU specs
        this->batch_size = 1<<20;
    } else {
        this->batch_size = batch_size;
    }

    contexts.reserve(DeviceManager::get_device_count());
    
    DeviceManager::foreach_device(
        [this](const int& device_id, const cudaStream_t& stream) {
            contexts.emplace_back(
                stream,
                RenderingWorkspace(device_id),
                SceneWorkspace(device_id),
                this->batch_size
            );
        }
    );
}

void NeRFRenderingController::cancel() {
    if (request != nullptr) {
        if (request->is_canceled())
            return;
        else
            request->cancel();
    }
}

void NeRFRenderingController::submit(
    std::shared_ptr<RenderRequest> request
) {
    // TODO: batching/chunking/distributing requests across multiple GPUs
    const int device_id = 0;
    auto& ctx = contexts[device_id];

    ctx.min_step_size = min_step_size;

    std::vector<NeRFRenderable> renderables;
    renderables.reserve(request->renderables.size());

    for (auto& renderable : request->renderables) {
        NeRFProxy* proxy = renderable.proxy;
        if (proxy->can_render && proxy->is_visible && !proxy->should_destroy) {
            proxy->update_dataset_if_necessary(ctx.stream);
            renderables.push_back(renderable);
        }
    }

    renderables.shrink_to_fit();

    // this is not great but it will work for now...
    request->renderables = renderables;

    // TODO: Allow NeRFs and Datasets to be rendered independently
    if (renderables.size() == 0) {
        request->on_complete();
        return;
    }
    
    this->request = request;

    // split this request into batches
    uint32_t n_rays_per_batch = batch_size / 8;
    uint32_t n_rays_per_preview = 1<<14;

    unique_ptr<RenderTaskFactory> factory(
        create_render_task_factory(
            pattern,
            n_rays_per_batch,
            n_rays_per_preview
        )
    );

    this->tasks = factory->create_tasks(request.get());

    // get actual number of rays per batch
    int n_rays_max = 0;
    for (auto& task : tasks) {
        if (task.n_rays > n_rays_max)
            n_rays_max = task.n_rays;
    }

    // prepare for rendering and dispatch tasks
    bool requested_preview = (request->flags & RenderFlags::Preview) == RenderFlags::Preview;
    bool requested_final = (request->flags & RenderFlags::Final) == RenderFlags::Final;

    renderer.prepare_for_rendering(ctx, request->camera, renderables, n_rays_max, requested_final);

    // preview task cannot be canceled
    if (factory->can_preview() && requested_preview) {
        auto& preview_task = tasks[0];
        renderer.perform_task(ctx, preview_task);
        renderer.write_to_target(ctx, preview_task, request->output);
        request->on_progress(1.0f / (float)tasks.size());
    }

    // final_tasks are all other tasks after the optional first one
    if (requested_final) {
        const int final_tasks_start = factory->can_preview() ? 1 : 0;
        
        for (int i = final_tasks_start; i < tasks.size(); ++i) {
            if (request->is_canceled()) {
                request->on_cancel();
                return;
            }

            auto& task = tasks[i];
            renderer.perform_task(ctx, task);
            renderer.write_to_target(ctx, task, request->output);
            request->on_progress((float)i / (float)tasks.size());
        }
    }

    request->on_complete();
}

std::vector<size_t> NeRFRenderingController::get_cuda_memory_allocated() const {

    int n_gpus = DeviceManager::get_device_count();
    std::vector<size_t> sizes(n_gpus);

    // one context per GPU
    int i = 0;
    for (const auto& ctx : contexts) {
        size_t total = 0;
        
        total += ctx.render_ws.get_bytes_allocated();
        total += ctx.scene_ws.get_bytes_allocated();

        sizes[i++] = total;
    }

    return sizes;
}

TURBO_NAMESPACE_END
