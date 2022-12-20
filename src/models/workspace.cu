#include "../common.h"

#include "workspace.cuh"

using namespace nrc;

void Workspace::enlarge(cudaStream_t stream, uint32_t n_rays) {
	auto data = tcnn::allocate_workspace_and_distribute<Ray>(stream, &arena_allocation, (size_t)n_rays);
	rays = std::get<0>(data);
}
