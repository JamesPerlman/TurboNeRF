#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include "../common.h"


NRC_NAMESPACE_BEGIN

void print_matrix(const Eigen::Matrix4f& mat) {
    // Set floating point output precision to 2 decimal places
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(5) << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

inline NRC_HOST_DEVICE Eigen::Matrix4f nerf_to_nrc(Eigen::Matrix4f nerf_matrix)
{
    Eigen::Matrix4f result(nerf_matrix);
    result.col(1).array() *= -1;
    result.col(2).array() *= -1;
    // cycle axes xyz->zxy
    const Eigen::Matrix4f tmp = result.eval();
    result.row(2) = tmp.row(0);
    result.row(0) = tmp.row(1);
    result.row(1) = tmp.row(2);
    
    printf("INPUT: \n");
    print_matrix(nerf_matrix);

    printf("RESULT: \n");
    print_matrix(result);

    return result;
}

#include "../common.h"

NRC_NAMESPACE_END
