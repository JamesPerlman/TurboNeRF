#pragma once

#include "../common.h"
#include "workspace.cuh"

NRC_NAMESPACE_BEGIN

struct CascadedOccupancyGridWorkspace: Workspace {
	uint32_t n_total_elements;
    uint32_t n_bitfield_elements;

    uint8_t* bitfield;
    float* values;

    void enlarge(
        const cudaStream_t& stream,
        const uint32_t& n_total_elements,
        const bool& use_full_precision_values
    ) {
        free_allocations();

        this->n_bitfield_elements = (n_total_elements + 7) / 8;
        bitfield = allocate<uint8_t>(stream, n_bitfield_elements);

        if (use_full_precision_values) {
            this->n_total_elements = n_total_elements;
            values = allocate<float>(stream, n_total_elements);
        }
    }
};

NRC_NAMESPACE_END
