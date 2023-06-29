#include "../common.h"
#include "nerf.cuh"
#include "nerf-proxy.cuh"

TURBO_NAMESPACE_BEGIN

NeRF::NeRF(const int& device_id, NeRFProxy* proxy)
    : device_id(device_id)
    , proxy(proxy)
    , dataset_ws(device_id)
    , params(device_id)
    , occupancy_grid(device_id, 128)
{ };

void NeRF::free_device_memory() {
    dataset_ws.free_allocations();
    params.free_allocations();
    occupancy_grid.free_device_memory();
}

TURBO_NAMESPACE_END
