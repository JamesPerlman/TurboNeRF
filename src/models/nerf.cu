#include "../common.h"
#include "nerf.cuh"
#include "nerf-proxy.cuh"

TURBO_NAMESPACE_BEGIN

NeRF::NeRF(const int& device_id, const NeRFProxy* proxy)
    : device_id(device_id)
    , proxy(proxy)
    , dataset_ws(device_id)
    , params(device_id)
    , occupancy_grid(device_id, OccupancyGrid::get_max_n_levels(proxy->bounding_box.size()), 128)
{ };

TURBO_NAMESPACE_END
