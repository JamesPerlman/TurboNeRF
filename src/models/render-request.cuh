#pragma once

#include <functional>
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
    std::function<void(bool)> on_result;
    std::function<bool()> should_cancel;

    RenderRequest(
        const Camera& camera,
        std::vector<NeRFProxy*>& proxies,
        RenderTarget* output,
        std::function<void(bool)> on_result = nullptr,
        std::function<bool()> should_cancel = nullptr
    )
        : camera(camera)
        , proxies(proxies)
        , output(output)
        , on_result(on_result)
        , should_cancel(should_cancel)
    { };
};

NRC_NAMESPACE_END
