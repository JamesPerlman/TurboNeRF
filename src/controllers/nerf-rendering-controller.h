#pragma once

#include <cuda_runtime.h>

#include "../common.h"
#include "../models/render-buffer.cuh"
#include "../models/render-request.cuh"

NRC_NAMESPACE_BEGIN

struct NeRFRenderingController {

    NeRFRenderingController() = default;

    void request_render(
        // TODO: remove stream
        const cudaStream_t& stream,
        const RenderRequest& request,
        RenderBuffer& output
    );

};

NRC_NAMESPACE_END
