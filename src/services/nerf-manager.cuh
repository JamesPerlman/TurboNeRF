#pragma once

#include <memory>
#include <stdint.h>
#include <vector>

#include "../common.h"
#include "../core/nerf-network.cuh"
#include "../core/occupancy-grid.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../services/device-manager.cuh"

#include <optional>

TURBO_NAMESPACE_BEGIN

struct NeRFManager {
private:
	std::vector<NeRFProxy> proxies;

public:
	// TODO: protect nerfs with const getter?
	// There are downstream effects which make this impossible for now
	// like the fact that Workspaces are stored within the NeRF
	std::vector<NeRFProxy*> get_proxies() {
		std::vector<NeRFProxy*> proxy_ptrs;
		proxy_ptrs.reserve(proxies.size());
		for (auto& proxy : proxies) {
			proxy_ptrs.emplace_back(&proxy);
		}
		return proxy_ptrs;
	}

	// create a new nerf
	NeRFProxy* create(
		const BoundingBox& bbox
	) {
		proxies.emplace_back();

		NeRFProxy& proxy = proxies.back();
		proxy.nerfs.reserve(DeviceManager::get_device_count());

		DeviceManager::foreach_device([&](const int& device_id, const cudaStream_t& stream) {
			proxy.nerfs.emplace_back(device_id, bbox);
			NeRF& nerf = proxy.nerfs.back();
		});

		return &proxy;
	}

	// manage nerfs

	// destroy nerfs

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
