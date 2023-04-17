#pragma once
#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>
#include <vector>

#include "../models/render-task.cuh"
#include "../workspaces/rendering-workspace.cuh"
#include "../workspaces/scene-workspace.cuh"
#include "../common.h"

TURBO_NAMESPACE_BEGIN

struct Renderer {
    struct Context {
        const cudaStream_t& stream;
        RenderingWorkspace render_ws;
        SceneWorkspace scene_ws;
        NerfNetwork network;

        const uint32_t batch_size;

        Context(
            const cudaStream_t& stream,
            RenderingWorkspace render_ws,
            SceneWorkspace scene_ws,
            NerfNetwork network,
            uint32_t batch_size
        )
            : stream(stream)
            , render_ws(std::move(render_ws))
            , scene_ws(std::move(scene_ws))
            , network(std::move(network))
            , batch_size(std::move(batch_size))
        {};
    };

    void perform_task(
        Context& ctx,
        RenderTask& task
    );

    void write_to_target(
        Context& ctx,
        RenderTask& task,
        RenderTarget* target
    );

    void prepare_for_rendering(
        Context& ctx,
        const Camera& camera,
        const std::vector<NeRF*>& nerfs,
        const uint32_t& n_rays
    );
};

TURBO_NAMESPACE_END
