#pragma once

#include <cuda_runtime.h>

#include "../common.h"
#include "../models/render-buffer.cuh"
#include "../models/render-request.cuh"
#include "../models/rendering-workspace.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFRenderingController {

    NeRFRenderingController(
        uint32_t batch_size = 0
    );

    void request_render(
        // TODO: remove stream
        const cudaStream_t& stream,
        const RenderRequest& request
    );

private:
    
    uint32_t batch_size;

    RenderingWorkspace workspace;

};

NRC_NAMESPACE_END
