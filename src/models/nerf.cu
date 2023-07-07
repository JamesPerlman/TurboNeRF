#include "../common.h"
#include "nerf.cuh"
#include "nerf-proxy.cuh"

TURBO_NAMESPACE_BEGIN

NeRF::NeRF(const int& device_id, NeRFProxy* proxy)
    : device_id(device_id)
    , dataset_ws(device_id)
    , params(device_id)
    , occupancy_grid(device_id, 128)
    , network(device_id)
    , proxy(proxy)
{ };

void NeRF::free_device_memory() {
    dataset_ws.free_allocations();
    params.free_allocations();
    occupancy_grid.free_device_memory();
}

void NeRF::free_training_data() {
    network.free_training_data();
    dataset_ws.free_allocations();
    is_image_data_loaded = false;
}

TURBO_NAMESPACE_END
