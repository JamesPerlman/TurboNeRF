#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include "common.h"
#include "main.h"
#include "models/dataset.hpp"


// Declare a global device function
__global__ void helloCuda(void)
{
    printf("Hello World from GPU!!\n");
}

int main()
{
	nrc::Dataset dataset = nrc::Dataset::from_file("E:\\2022\\nerf-library\\FascinatedByFungi2022\\hydnellum-peckii-cluster\\transforms.json");
    dataset.load_images_in_parallel();
    
    // Launch the "helloCuda" kernel on the device
    helloCuda<<<1, 1>>> ();

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    return 0;
}
