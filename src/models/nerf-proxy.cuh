#pragma once

#include <optional>
#include <vector>

#include "../common.h"
#include "dataset.h"
#include "nerf.cuh"

NRC_NAMESPACE_BEGIN

/**
 * A NeRFProxy is a reference to one or more NeRF objects.
 * The NeRFs should all share copies of the same data,
 * but they may be distributed across multiple GPUs.
 */

struct NeRFProxy {
    std::vector<NeRF> nerfs;
    bool visible = true;
    
    // TODO:
    // transform
    // bounding box (training, rendering)
    // masks
    // distortions

    BoundingBox get_bounding_box() const {
        return nerfs[0].bounding_box;
    }

    std::vector<NeRF*> get_nerf_ptrs() {
        std::vector<NeRF*> ptrs;
        ptrs.reserve(nerfs.size());
        for (auto& nerf : nerfs) {
            ptrs.emplace_back(&nerf);
        }
        return ptrs;
    }
};

NRC_NAMESPACE_END
