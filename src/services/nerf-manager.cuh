#pragma once

#include <fstream>
#include <iostream>
#include <json/json.hpp>
#include <map>
#include <memory>
#include <optional>
#include <stdint.h>
#include <vector>

#include "../core/nerf-network.cuh"
#include "../core/occupancy-grid.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../services/device-manager.cuh"
#include "../utils/nerf-constants.cuh"
#include "../utils/observable.cuh"
#include "../utils/parallel-utils.cuh"
#include "../common.h"
#include "file-manager.cuh"

#define NERF_SNAPSHOT_VERSION 0

TURBO_NAMESPACE_BEGIN

struct NeRFManager {
private:

    std::vector<NeRFProxy> _proxies;

    std::pair<int, NeRFProxy*> find_first_unused_proxy() {
        for (int i = 0; i < _proxies.size(); ++i) {
            auto& proxy = _proxies[i];
            if (!proxy.is_valid) {
                return { i, &proxy };
            }
        }
        throw std::runtime_error("No more proxies available");
    }


public:

    static int n_max_nerfs() {
        return NeRFConstants::n_max_nerfs;
    }

    NeRFManager() {
        _proxies.resize(NeRFManager::n_max_nerfs());
    }

    // TODO: protect nerfs with const getter?
    // There are downstream effects which make this impossible for now
    // like the fact that Workspaces are stored within the NeRF
    NeRFProxy* proxy_for_id(const int& id) {
        if (id < 0 || id >= _proxies.size()) {
            return nullptr;
        }
        // we can guarantee that the id is the index for now
        return &_proxies[id];
    }

    int n_proxies() const {
        return _proxies.size();
    }

    // create a new nerf
    NeRFProxy* create() {
        auto item = find_first_unused_proxy();
        int index = item.first;
        NeRFProxy* proxy = item.second;
        
        proxy->id = index;
        proxy->nerfs.clear();
        proxy->nerfs.reserve(DeviceManager::get_device_count());

        DeviceManager::foreach_device([&](const int& device_id, const cudaStream_t& stream) {
            proxy->nerfs.emplace_back(device_id, proxy);
        });

        proxy->is_valid = true;
        proxy->can_render = false;
        proxy->training_step = 0;

        return proxy;
    }

    NeRFProxy* clone(NeRFProxy* proxy) {
        auto item = find_first_unused_proxy();
        int index = item.first;
        NeRFProxy* new_proxy = item.second;

        // the dataset is the only part that should not be cloned
        new_proxy->dataset = std::nullopt;

        // only clear the nerfs if there are no other nerfs with this id
        new_proxy->nerfs.clear();
        new_proxy->nerfs.reserve(DeviceManager::get_device_count());
        new_proxy->render_bbox = proxy->render_bbox;
        new_proxy->training_bbox = proxy->training_bbox;
        new_proxy->transform = proxy->transform;
        new_proxy->id = index;

        DeviceManager::foreach_device([&](const int& device_id, const cudaStream_t& stream) {
            new_proxy->nerfs.emplace_back(proxy->nerfs[device_id]);
            NeRF& nerf = new_proxy->nerfs.back();
            nerf.proxy = new_proxy;
        });

        new_proxy->is_valid = true;
        new_proxy->can_render = true;

        if (proxy->clone_source == nullptr) {
            new_proxy->clone_source = proxy;
        } else {
            new_proxy->clone_source = proxy->clone_source;
        }

        return new_proxy;
    }

    // destroy nerfs

    void destroy(NeRFProxy* proxy) {
        
        // need to check for duplicates before clearing nerfs
        bool has_clones = proxy_has_clones(proxy);

        proxy->id = -1;
        proxy->is_valid = false;
        proxy->can_render = false;
        proxy->clone_source = nullptr;

        if (proxy->dataset.has_value()) {
            proxy->dataset->unload_images();
            proxy->dataset.reset();
        }

        if (!has_clones) {
            proxy->free_device_memory();
        }
    }

    bool proxy_has_clones(const NeRFProxy* proxy) {
        for (int i = 0; i < _proxies.size(); ++i) {
            const auto& other_proxy = _proxies[i];
            
            if (!other_proxy.is_valid) {
                continue;
            }

            if (proxy->clone_source == &other_proxy || other_proxy.clone_source == proxy) {
                return true;
            }
        }
        return false;
    }

    // copy between GPUs?

    std::vector<size_t> get_cuda_memory_allocated() const {
        const int n_gpus = DeviceManager::get_device_count();
        
        std::vector<size_t> sizes(n_gpus, 0);

        for (const auto& proxy : _proxies) {
            int i = 0;
            // one nerf per gpu
            for (const auto& nerf : proxy.nerfs) {
                size_t total = 0;
                
                total += nerf.params.get_bytes_allocated();
                total += nerf.occupancy_grid.workspace.get_bytes_allocated();

                sizes[i++] += total;
            }
        }

        return sizes;
    }
};

TURBO_NAMESPACE_END
