#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>

#include "../common.h"
#include "linalg.cuh"

NRC_NAMESPACE_BEGIN

inline NRC_HOST_DEVICE Matrix4f nerf_to_nrc(Matrix4f nerf_matrix)
{
    Matrix4f result = nerf_matrix;
    // invert column 1
    result.m01 = -nerf_matrix.m01;
    result.m11 = -nerf_matrix.m11;
    result.m21 = -nerf_matrix.m21;
    result.m31 = -nerf_matrix.m31;

    // invert column 2
    result.m02 = -nerf_matrix.m02;
    result.m12 = -nerf_matrix.m12;
    result.m22 = -nerf_matrix.m22;
    result.m32 = -nerf_matrix.m32;

    // roll axes xyz -> yzx
    const Matrix4f tmp = result;
    // x -> y
    result.m00 = tmp.m10; result.m01 = tmp.m11; result.m02 = tmp.m12; result.m03 = tmp.m13;
    // y -> z
    result.m10 = tmp.m20; result.m11 = tmp.m21; result.m12 = tmp.m22; result.m13 = tmp.m23;
    // z -> x
    result.m20 = tmp.m00; result.m21 = tmp.m01; result.m22 = tmp.m02; result.m23 = tmp.m03;
    
    return result;
}

NRC_NAMESPACE_END
