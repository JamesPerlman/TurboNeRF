#pragma once

#include <vector>
#include <memory>

#include "nerf-proxy.cuh"
#include "../common.h"
#include "../effects/spatial-effect.cuh"

TURBO_NAMESPACE_BEGIN

class NeRFRenderable {
    public:

    NeRFProxy* proxy;

    std::vector<std::shared_ptr<ISpatialEffect>> spatial_effects;

    BoundingBox bounding_box;

    NeRFRenderable(NeRFProxy* proxy, std::vector<std::shared_ptr<ISpatialEffect>> spatial_effects = {})
        : proxy(proxy)
        , spatial_effects(spatial_effects)
    {
        BoundingBox bbox = proxy->render_bbox.get();
        for (const auto& effect : spatial_effects) {
            bbox = effect->get_bbox(bbox);
        }
        bounding_box = bbox;
    };
    
    // todo: masks
};

TURBO_NAMESPACE_END
