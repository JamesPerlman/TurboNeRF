#pragma once

#include <stdint.h>

#include "../common.h"
#include "camera.cuh"
#include "nerf.cuh"
#include "render-buffer.cuh"

NRC_NAMESPACE_BEGIN

struct RenderRequest {
    RenderBuffer destination;
    Camera camera;
    const std::vector<const NeRF*> nerfs;

    RenderRequest(RenderBuffer destination, Camera camera, const std::vector<const NeRF*> nerfs)
        : destination(destination), camera(camera), nerfs(nerfs) {};
};

NRC_NAMESPACE_END
