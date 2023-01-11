#pragma once
#include "../common.h"

NRC_NAMESPACE_BEGIN

/**
 * Calculates block counts and offsets, returns the number of elements to compact.
 */
int calculate_block_counts_and_offsets(
    const cudaStream_t& stream,
    const int n_elements,
    const int block_size,
    const bool* d_predicate,
    int* d_block_counts,
    int* d_block_offsets
);

/**
 * Takes in calculated block counts and offsets, writes the indices of elements to compact to d_output.
 */
void generate_compaction_indices(
    const cudaStream_t& stream,
    const int n_elements,
    const int block_size,
    const bool* d_predicate,
    const int* d_block_offsets,
    int* d_output
);

NRC_NAMESPACE_END
