#include "stream-compaction.cuh"
#include <thrust/count.h>

TURBO_NAMESPACE_BEGIN

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

template <typename T>
struct nonzero {
    __host__ __device__
    bool operator()(const T& x) const {
        return x != 0;
    }
};

size_t count_nonzero_elements(
    const cudaStream_t& stream,
    const size_t n_elements,
    const uint32_t* predicate
) {
    thrust::device_ptr<const uint32_t> predicate_ptr(predicate);
    return thrust::count_if(
        thrust::cuda::par.on(stream),
        predicate_ptr,
        predicate_ptr + n_elements,
        nonzero<uint32_t>()
    );
};

void generate_nonzero_compaction_indices(
    const cudaStream_t& stream,
    const size_t n_elements,
    const uint32_t* values,
    int* indices
) {
    thrust::device_ptr<const uint32_t> values_ptr(values);
    thrust::device_ptr<int> indices_ptr(indices);

    thrust::counting_iterator<int> counting(0);
    thrust::copy_if(
        thrust::cuda::par.on(stream),
        counting,
        counting + n_elements,
        values_ptr,
        indices_ptr,
        nonzero<uint32_t>()
    );
}

TURBO_NAMESPACE_END
