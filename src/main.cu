#include <iostream>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include "common.h"
#include "main.h"
#include "json-bindings/eigen-json.hpp"


// Declare a global device function
__global__ void helloCuda(void)
{
    printf("Hello World from GPU!!\n");
}

int main()
{
    Eigen::Matrix4f e = nrc::from_json({});
    // Launch the "helloCuda" kernel on the device
    helloCuda<<<1, 1>>> ();

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    return 0;
}
