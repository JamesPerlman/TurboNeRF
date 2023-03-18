#pragma once

#include "../common.h"
#include "workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct OccupancyGridWorkspace: Workspace {

    using Workspace::Workspace;

	uint32_t n_total_elements;
    uint32_t n_bitfield_elements;

    uint8_t* bitfield;
    uint8_t* bitcounts;
    float* values;

    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_levels,
        const uint32_t& n_elements_per_level,
        const bool& is_trainable
    ) {
        free_allocations();

        this->n_bitfield_elements = n_elements_per_level;
        this->n_total_elements = n_levels * n_elements_per_level;

        bitfield = allocate<uint8_t>(stream, n_bitfield_elements);

        if (is_trainable) {
            values = allocate<float>(stream, n_total_elements);
            bitcounts = allocate<uint8_t>(stream, n_bitfield_elements);
        }
    }
};

TURBO_NAMESPACE_END
