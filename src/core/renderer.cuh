#pragma once
#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../models/render-buffer.cuh"
#include "../models/render-request.cuh"
#include "../workspaces/rendering-workspace.cuh"

NRC_NAMESPACE_BEGIN

struct Renderer {
    struct Context {
        const cudaStream_t& stream;
        RenderingWorkspace workspace;

        const uint32_t batch_size;

        Context(
            const cudaStream_t& stream,
            RenderingWorkspace& workspace,
            const uint32_t& batch_size
        )
            : stream(stream)
            , workspace(workspace)
            , batch_size(batch_size)
        {};
    };

    void render(Context& ctx, RenderRequest& request);

private:
    uint32_t render_area = 0;

    void enlarge_workspace_if_needed(Context& ctx, const RenderRequest& request);
};

NRC_NAMESPACE_END
