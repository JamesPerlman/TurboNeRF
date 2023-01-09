#pragma once

#include <stdint.h>

#include "../common.h"
#include "camera.cuh"

NRC_NAMESPACE_BEGIN

struct RenderRequest {
    uint32_t width;
    uint32_t height;
    Camera camera;
    
};

NRC_NAMESPACE_END
