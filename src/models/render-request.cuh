#pragma once

#include <stdint.h>

#include "../common.h"
#include "camera.cuh"
#include "nerf.cuh"
#include "render-buffer.cuh"

NRC_NAMESPACE_BEGIN

struct RenderRequest {
    RenderBuffer output;
    Camera camera;
    std::vector<NeRF*> nerfs;

    RenderRequest(RenderBuffer output, Camera camera, std::vector<NeRF*> nerfs)
        : output(output), camera(camera), nerfs(nerfs) {};
};

NRC_NAMESPACE_END
