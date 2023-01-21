#pragma once

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

NRC_NAMESPACE_END
