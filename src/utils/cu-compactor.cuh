#pragma once
#include "../common.h"

NRC_NAMESPACE_BEGIN
/**
 * This function takes in a predicate of bools,
 * and writes the indices of the true values to d_output.
 */
int generate_compaction_indices(
    const cudaStream_t& stream,
    const int n_elements,
    const int blockSize,
    const bool* d_predicate,
    int* d_output
);

NRC_NAMESPACE_END
