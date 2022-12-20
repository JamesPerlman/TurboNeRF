#pragma once

#include "../common.h"

#include "ray.hpp"

#include <tiny-cuda-nn/common_device.h>

NRC_NAMESPACE_BEGIN

// NeRFWorkspace?
struct Workspace {
private:
	tcnn::GPUMemoryArena::Allocation arena_allocation;
	
public:
	Workspace() : arena_allocation() {};
	Ray* rays;
	
	void enlarge(cudaStream_t stream, uint32_t n_rays);
};

NRC_NAMESPACE_END
