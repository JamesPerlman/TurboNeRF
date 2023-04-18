#pragma once

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include "../common.h"
#include "parallel-utils.cuh"

TURBO_NAMESPACE_BEGIN

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

template <typename T>
size_t count_valued_elements(
    const cudaStream_t& stream,
    const size_t n_elements,
    const T* elements,
    const T& value
) {
    thrust::device_ptr<const T> elems_ptr(elements);
    return thrust::count_if(
        MAKE_EXEC_POLICY(stream),
        elems_ptr,
        elems_ptr + n_elements,
        thrust::detail::equal_to_value<const T>(value)
    );
};

template <typename T>
void generate_valued_compaction_indices(
    const cudaStream_t& stream,
    const size_t n_elements,
    const T* elements,
    const T& value,
    int* indices
) {
    thrust::device_ptr<const T> elems_ptr(elements);
    thrust::device_ptr<int> indices_ptr(indices);

    thrust::counting_iterator<int> counting(0);
    thrust::copy_if(
        MAKE_EXEC_POLICY(stream),
        counting,
        counting + n_elements,
        elems_ptr,
        indices_ptr,
        thrust::detail::equal_to_value<const T>(value)
    );
}

TURBO_NAMESPACE_END
