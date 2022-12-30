#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/sort.h>


#include "../common.h"

NRC_NAMESPACE_BEGIN

/**
 * @brief Find the index of the last item in a buffer of presorted ascending elements whose value is less than a given `max_value`
 * 
 * @param stream A cudaStream_t to run all operations on
 * @param data_begin_ptr A thrust::device_ptr that points to the first element in the buffer
 * @param n_elements The number of elements stored in the buffer
 * @param max_value The maximum value to compare each element against
 * @return * The index of the last element less than `max_value`, or -1 if no element was found.
 */
template <typename T>
inline NRC_HOST_DEVICE int find_last_lt_presorted(
	const cudaStream_t& stream,
    const thrust::device_ptr<T>& data_begin_ptr,
	const size_t& n_elements,
    const T& max_value
) {
	auto exec_policy = thrust::cuda::par_nosync.on(stream);

	auto iter_begin_reverse = thrust::make_reverse_iterator(data_begin_ptr);
	auto iter_end_reverse = thrust::make_reverse_iterator(data_begin_ptr + n_elements);

    // find the last element in the vector that is less than the maximum value
    auto last = thrust::find_if(
		exec_policy,
        iter_end_reverse,
        iter_begin_reverse,
        [max_value] NRC_HOST_DEVICE (T x) { return x < max_value; }
    );

    // if an element was found, compute its index relative to the beginning of the vector
    if (last != iter_begin_reverse)
    {
        return static_cast<int>(thrust::distance(data_begin_ptr, last.base() - 1));
    }

    // otherwise, return -1 to indicate that no element was found
    return -1;
}

NRC_NAMESPACE_END
