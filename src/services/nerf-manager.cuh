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
#include "../utils/parallel-utils.cuh"
#include "../common.h"
#include "file-manager.cuh"

#define NERF_SNAPSHOT_VERSION 0

TURBO_NAMESPACE_BEGIN

using proxy_id_t = uint32_t;

struct NeRFManager {
private:
	std::map<proxy_id_t, NeRFProxy> proxies;
	proxy_id_t max_id = 0;

	proxy_id_t add_empty_proxy() {
		proxy_id_t proxy_id = max_id;
		proxies[proxy_id] = NeRFProxy();
		max_id++;
		return proxy_id;
	}

public:
	// TODO: protect nerfs with const getter?
	// There are downstream effects which make this impossible for now
	// like the fact that Workspaces are stored within the NeRF
	std::vector<NeRFProxy*> get_proxies() {
		std::vector<NeRFProxy*> proxy_ptrs;
		proxy_ptrs.reserve(proxies.size());
		// enumerate through proxies
		for (auto& [id, proxy] : proxies) {
			proxy_ptrs.push_back(&proxy);
		}

		return proxy_ptrs;
	}

	NeRFProxy& get_proxy(const uint32_t& proxy_id) {
		return proxies[proxy_id];
	}

	NeRFProxy* get_proxy_ptr(const uint32_t& proxy_id) {
		return &proxies[proxy_id];
	}

	// create a new nerf
	proxy_id_t create(
		const Dataset& dataset
	) {
		proxy_id_t proxy_id = add_empty_proxy();
		auto& proxy = proxies[proxy_id];

		proxy.dataset = dataset;
		proxy.nerfs.reserve(DeviceManager::get_device_count());

		DeviceManager::foreach_device([&](const int& device_id, const cudaStream_t& stream) {
			proxy.nerfs.emplace_back(device_id, dataset.bounding_box);
		});

		return proxy_id;
	}

	void save(const proxy_id_t& proxy_id, const std::string& path) const {
		const NeRFProxy& proxy = proxies.at(proxy_id);
		FileManager::save(proxy, path);
	}

	proxy_id_t load(const std::string& path) {
		proxy_id_t proxy_id = add_empty_proxy();
		auto& proxy = proxies[proxy_id];
		FileManager::load(proxy, path);
		return proxy_id;
	}

	// destroy nerfs

	// copy between GPUs?	

	std::vector<size_t> get_cuda_memory_allocated() const {
		const int n_gpus = DeviceManager::get_device_count();
		
		std::vector<size_t> sizes(n_gpus, 0);

		for (const auto& [id, proxy] : proxies) {
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
