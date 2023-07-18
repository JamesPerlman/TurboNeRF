#pragma once

#include <optional>
#include <vector>

#include "../models/updatable-property.cuh"
#include "../math/transform4f.cuh"
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

    int id = -1;
    
    // nerf props
    UpdatableProperty<BoundingBox> training_bbox = BoundingBox();
    UpdatableProperty<BoundingBox> render_bbox = BoundingBox();
    UpdatableProperty<Transform4f> transform = Transform4f::Identity();

    bool is_valid = false;
    bool can_render = false;
    bool is_dataset_dirty = true;
    bool is_visible = true;
    uint32_t n_appearances = 0;
    uint32_t training_step = 0;

    // runloop flags (these are only used by the Blender bridge)
    // TODO: move these to a separate struct (maybe a wrapper?)
    bool should_destroy = false;
    bool should_reset = false;
    bool should_train = false;
    bool should_free_training_data = false;
    NeRFProxy* clone_source = nullptr;

    // constructor
    NeRFProxy() = default;
    
    // TODO:
    // bounding box (training, rendering)
    // masks
    // distortions

    std::vector<NeRF*> get_nerf_ptrs() {
        std::vector<NeRF*> ptrs;
        ptrs.reserve(nerfs.size());
        for (auto& nerf : nerfs) {
            ptrs.emplace_back(&nerf);
        }
        return ptrs;
    }

    bool is_dirty() const {
        return is_dataset_dirty || training_bbox.is_dirty() || render_bbox.is_dirty() || transform.is_dirty();
    }

    void update_dataset_if_necessary(const cudaStream_t& stream) {
        if (!is_dataset_dirty) {
            return;
        }

        is_dataset_dirty = false;

        if (!dataset.has_value()) {
            return;
        }

        // supported changes: Everything about Cameras, except how many there are.
        // aka the number of cameras must remain the same as when the nerf_proxy was constructed.

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
    }

    void free_device_memory() {
        for (auto& nerf : nerfs) {
            nerf.free_device_memory();
        }
    }

    void attach_dataset(const Dataset& dataset) {
        this->dataset = dataset;
        this->render_bbox = dataset.bounding_box;
        this->training_bbox = dataset.bounding_box;
        this->is_dataset_dirty = true;
        this->n_appearances = dataset.images.size();
    }

    void detach_dataset() {
        this->dataset.reset();
        this->is_dataset_dirty = true;
    }

    bool can_train() const {
        if (!is_valid || !dataset.has_value()) {
            return false;
        }

        for (const auto& nerf : nerfs) {
            if (!nerf.network.can_train() || !nerf.is_image_data_loaded) {
                return false;
            }
        }
        
        return true;
    }

    bool is_image_data_loaded() const {
        if (!can_train()) {
            return false;
        }

        for (const auto& nerf : nerfs) {
            if (!nerf.is_image_data_loaded) {
                return false;
            }
        }

        return true;
    }
};

TURBO_NAMESPACE_END
