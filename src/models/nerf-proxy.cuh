#pragma once

#include <optional>
#include <vector>

#include "../common.h"
#include "dataset.h"
#include "nerf.cuh"

TURBO_NAMESPACE_BEGIN

/**
 * A NeRFProxy is a reference to one or more NeRF objects.
 * The NeRFs should all share copies of the same data,
 * but they may be distributed across multiple GPUs.
 */

struct NeRFProxy {
    std::vector<NeRF> nerfs;
    std::optional<Dataset> dataset;
    
    bool is_visible = true;
    bool is_dataset_dirty = true;

    NeRFProxy() = default;
    
    // TODO:
    // transform
    // bounding box (training, rendering)
    // masks
    // distortions

    BoundingBox get_bounding_box() const {
        if (dataset.has_value()) {
            return dataset->bounding_box;
        }

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

    void update_dataset_if_necessary(const cudaStream_t& stream) {
        // supported changes: Everything about Cameras, except how many there are.
        // aka the number of cameras must remain the same as when the nerf_proxy was constructed.
        if (!is_dataset_dirty) {
            return;
        }

        // TODO: do for all NeRFs
        auto& nerf = nerfs[0];
        
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                nerf.dataset_ws.cameras,
                dataset->cameras.data(),
                dataset->cameras.size() * sizeof(Camera),
                cudaMemcpyHostToDevice,
                stream
            )
        );

        is_dataset_dirty = false;
    }
};

TURBO_NAMESPACE_END
