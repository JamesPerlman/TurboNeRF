#pragma once

#include <optional>
#include <vector>

#include "../common.h"

#include "nerf.cuh"

NRC_NAMESPACE_BEGIN

/**
 * A NeRFProxy is a reference to one or more NeRF objects.
 * The NeRFs should all share copies of the same data,
 * but they may be distributed across multiple GPUs.
 */

struct NeRFProxy {
    std::vector<NeRF> nerfs;

    BoundingBox get_bounding_box() const {
        return nerfs[0].bounding_box;
    }
};

NRC_NAMESPACE_END
