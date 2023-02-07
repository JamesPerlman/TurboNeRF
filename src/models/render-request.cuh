#pragma once

#include <stdint.h>

#include "../common.h"
#include "camera.cuh"
#include "nerf-proxy.cuh"
#include "render-buffer.cuh"

NRC_NAMESPACE_BEGIN

struct RenderRequest {
    const Camera camera;
    std::vector<NeRFProxy*> proxies;
    RenderBuffer* output;

    RenderRequest(const Camera& camera, std::vector<NeRFProxy*>& proxies, RenderBuffer* output)
        : camera(camera), proxies(proxies), output(output) {};
};

NRC_NAMESPACE_END
