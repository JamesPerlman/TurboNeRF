#pragma once

#include "../../common.h"
#include "../workspaces/workspace.cuh"

TURBO_NAMESPACE_BEGIN

template <typename Params>
struct EffectWorkspace: public Workspace {
    Params* params;

    void allocate(const cudaStream_t& stream) {
        free_allocations();
        params = allocate<Params>(stream, 1);
    }
};

template <typename Params>
class SpatialEffect {
    public:

    EffectWorkspace<Params> workspace;

    SpatialEffect(const int device_id) : workspace(device_id) {};

    virtual __host__ void init(
        const cudaStream_t& stream,
        const Params& params
    ) {
        workspace.allocate(stream);
    };
    
    virtual __host__ void destroy() {
        workspace.free_allocations();
    };

    virtual __device__ void apply(
        const Params& params,
        const float& in_x, const float& in_y, const float& in_z,
        float& out_x, float& out_y, float& out_z
    ) = 0;
};

TURBO_NAMESPACE_END
