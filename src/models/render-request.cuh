#pragma once

#include <functional>
#include <stdint.h>

#include "../render-targets/render-target.cuh"
#include "../common.h"
#include "camera.cuh"
#include "nerf-proxy.cuh"

NRC_NAMESPACE_BEGIN

typedef std::function<void(bool)> OnResultCallback;
typedef std::function<bool()> ShouldCancelCallback;

struct RenderRequest {
    const Camera camera;
    std::vector<NeRFProxy*> proxies;
    RenderTarget* output;
    OnResultCallback on_result;
    ShouldCancelCallback should_cancel;

    RenderRequest(
        const Camera& camera,
        std::vector<NeRFProxy*>& proxies,
        RenderTarget* output,
        OnResultCallback on_result = nullptr,
        ShouldCancelCallback should_cancel = nullptr
    )
        : camera(camera)
        , proxies(proxies)
        , output(output)
        , on_result(on_result)
        , should_cancel(should_cancel)
    { };
};

NRC_NAMESPACE_END
