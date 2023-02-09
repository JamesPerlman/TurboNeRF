#pragma once

#include <stdint.h>

#include "../render-targets/render-target.cuh"
#include "../common.h"
#include "camera.cuh"
#include "nerf-proxy.cuh"

NRC_NAMESPACE_BEGIN

struct RenderRequest {
    const Camera camera;
    std::vector<NeRFProxy*> proxies;
    RenderTarget* output;

    RenderRequest(const Camera& camera, std::vector<NeRFProxy*>& proxies, RenderTarget* output)
        : camera(camera), proxies(proxies), output(output) {};
};

NRC_NAMESPACE_END
