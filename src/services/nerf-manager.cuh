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

NRC_NAMESPACE_BEGIN

struct NeRFManager {
private:
	std::vector<NeRFProxy> proxies;
	const int n_gpus = 1; // does nothing, for now
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
};

NRC_NAMESPACE_END
