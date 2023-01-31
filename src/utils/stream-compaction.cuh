#pragma once

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

size_t count_true_elements(
    const cudaStream_t& stream,
    const size_t n_elements,
    const bool* predicate
);

void generate_compaction_indices(
    const cudaStream_t& stream,
    const size_t n_elements,
    const bool* predicate,
    int* indices
);

size_t count_nonzero_elements(
    const cudaStream_t& stream,
    const size_t n_elements,
    const uint32_t* input
);

void generate_nonzero_compaction_indices(
    const cudaStream_t& stream,
    const size_t n_elements,
    const uint32_t* values,
    int* indices
);

NRC_NAMESPACE_END
