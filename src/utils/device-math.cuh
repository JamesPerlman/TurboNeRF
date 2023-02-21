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

NRC_NAMESPACE_END
