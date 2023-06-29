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
#include "../utils/parallel-utils.cuh"
#include "../common.h"
#include "file-manager.cuh"

#define NERF_SNAPSHOT_VERSION 0

TURBO_NAMESPACE_BEGIN

struct NeRFManager {
private:

	int n_proxies_created = 0;

	std::vector<NeRFProxy> proxies;

	NeRFProxy* find_first_unused_proxy() {
		for (auto& proxy : proxies) {
			if (!proxy.is_valid) {
				return &proxy;
			}
		}
		throw std::runtime_error("No more proxies available");
	}

public:

	NeRFManager() {
		proxies.resize(NeRFConstants::n_max_nerfs);
	}

	// TODO: protect nerfs with const getter?
	// There are downstream effects which make this impossible for now
	// like the fact that Workspaces are stored within the NeRF
	std::vector<NeRFProxy*> get_proxies() {
		std::vector<NeRFProxy*> proxy_ptrs;
		proxy_ptrs.reserve(proxies.size());
		// enumerate through proxies
		for (auto& proxy : proxies) {
			if (!proxy.is_valid) {
				continue;
			}
			proxy_ptrs.push_back(&proxy);
		}
		proxy_ptrs.shrink_to_fit();
		return proxy_ptrs;
	}

	// create a new nerf
	NeRFProxy* create() {
		NeRFProxy* proxy = find_first_unused_proxy();
		
		proxy->id = n_proxies_created++;
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

	NeRFProxy* clone(const NeRFProxy* proxy) {
		NeRFProxy* new_proxy = find_first_unused_proxy();

		// the dataset is the only part that should not be cloned
		new_proxy->dataset = std::nullopt;
		// only clear the nerfs if there are no other nerfs with this id
		new_proxy->nerfs.clear();
		new_proxy->nerfs.reserve(DeviceManager::get_device_count());
		new_proxy->render_bbox = proxy->render_bbox;
		new_proxy->training_bbox = proxy->training_bbox;
		new_proxy->transform = proxy->transform;
		new_proxy->id = proxy->id;

		DeviceManager::foreach_device([&](const int& device_id, const cudaStream_t& stream) {
			new_proxy->nerfs.emplace_back(proxy->nerfs[device_id]);
			NeRF& nerf = new_proxy->nerfs.back();
			nerf.proxy = new_proxy;
		});

		new_proxy->is_valid = true;
		new_proxy->can_render = true;
		new_proxy->can_train = false;

		return new_proxy;
	}

	// destroy nerfs

	void destroy(NeRFProxy* proxy) {
		
		proxy->is_valid = false;
		proxy->can_render = false;
		proxy->can_train = false;

		if (proxy->dataset.has_value()) {
			proxy->dataset->unload_images();
			proxy->dataset.reset();
		}

		// need to check for duplicates before clearing nerfs
		size_t n_nerfs_with_this_id = 0;
		
		for (const auto& other_proxy : proxies) {
			if (other_proxy.is_valid && other_proxy.id == proxy->id) {
				++n_nerfs_with_this_id;
			}
		}

		proxy->id = -1;

		if (n_nerfs_with_this_id == 0) {
			proxy->free_device_memory();
		}
	}

	// copy between GPUs?	

	std::vector<size_t> get_cuda_memory_allocated() const {
		const int n_gpus = DeviceManager::get_device_count();
		
		std::vector<size_t> sizes(n_gpus, 0);

		for (const auto& proxy : proxies) {
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
