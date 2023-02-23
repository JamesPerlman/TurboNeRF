#pragma once

#include "../common.h"

NRC_NAMESPACE_BEGIN

/**
 * Just a simple function to divide two integers
 * it seems to work fine for most applications.
 * An alternative approach: https://github.com/milakov/int_fastdiv 
 */

inline __device__ int divide(const int& a, const int& b) {
    return (int)(__fdividef((float)a, (float)b));
}

/**
 * A modulo function that sacrifices some accuracy for speed.
 * 
 */

inline __device__ int modulo(const int& a, const int& b) {
    return a - (b * divide(a, b));
}

/**
 * A little convenience bitwise setter function
 * 
 */

template <typename T, typename U>
inline __device__ __host__ T set_bit(const T& bitmask, const U bit, const bool value) {
    return value
        ? (bitmask |  bit)
        : (bitmask & ~bit);
}

NRC_NAMESPACE_END
