#include "stream-compaction.cuh"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

NRC_NAMESPACE_BEGIN

size_t count_true_elements(
    const cudaStream_t& stream,
    const size_t n_elements,
    const bool* predicate
) {
    thrust::device_ptr<const bool> predicate_ptr(predicate);
    return thrust::count_if(
        thrust::cuda::par.on(stream),
        predicate_ptr,
        predicate_ptr + n_elements,
        thrust::detail::equal_to_value<const bool>(true)
    );
}

void generate_compaction_indices(
    const cudaStream_t& stream,
    const size_t n_elements,
    const bool* predicate,
    int* indices
) {
    thrust::device_ptr<const bool> predicate_ptr(predicate);
    thrust::device_ptr<int> indices_ptr(indices);

    thrust::counting_iterator<int> counting(0);
    thrust::copy_if(
        thrust::cuda::par.on(stream),
        counting,
        counting + n_elements,
        predicate_ptr,
        indices_ptr,
        thrust::detail::equal_to_value<const bool>(true)
    );
}

NRC_NAMESPACE_END
