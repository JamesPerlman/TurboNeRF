#pragma once

#include "../common.h"

TURBO_NAMESPACE_BEGIN

inline NRC_HOST_DEVICE float get_dt(
    const float& t,
    const float& cone_angle,
    const float& dt_min,
    const float& dt_max
) {
    return tcnn::clamp(t * cone_angle, dt_min, dt_max);
}

TURBO_NAMESPACE_END
