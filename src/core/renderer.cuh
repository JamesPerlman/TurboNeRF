#pragma once
#include <cuda_runtime.h>
#include <tiny-cuda-nn/common.h>

#include "../models/render-request.cuh"
#include "../workspaces/rendering-workspace.cuh"
#include "../common.h"

NRC_NAMESPACE_BEGIN

struct Renderer {
    struct Context {
        const cudaStream_t& stream;
        RenderingWorkspace workspace;

        const uint32_t batch_size;

        Context(
            const cudaStream_t& stream,
            RenderingWorkspace workspace,
            uint32_t batch_size
        )
            : stream(stream)
            , workspace(std::move(workspace))
            , batch_size(std::move(batch_size))
        {};
    };

    void submit(
        Context& ctx,
        RenderRequest* request
    );

    void write_to(
        Context& ctx,
        RenderTarget* target
    );

private:

    void enlarge_workspace_if_needed(Context& ctx, RenderRequest* request);
};

NRC_NAMESPACE_END
