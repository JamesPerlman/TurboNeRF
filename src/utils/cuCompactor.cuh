#pragma once
#include "../common.h"

NRC_NAMESPACE_BEGIN

template <typename T, typename Predicate>
__global__ void computeBlockCounts(T *d_input, int length, int *d_BlockCounts, Predicate predicate);

template <typename T, typename Predicate>
__global__ void compactK(T *d_input, int length, T *d_output, int *d_BlocksOffset, Predicate predicate);

template <typename T, typename Predicate>
int compact(T *d_input, T *d_output, int length, Predicate predicate, int blockSize);

NRC_NAMESPACE_END