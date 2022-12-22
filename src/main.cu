#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <json/json.hpp>
#include <Eigen/Dense>

#include "common.h"
#include "main.h"
#include "models/dataset.h"
#include "controllers/nerf-training-controller.h"


// Declare a global device function
__global__ void helloCuda(void)
{
    printf("Hello World from GPU!!\n");
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
}

int main()
{
	nrc::Dataset dataset = nrc::Dataset::from_file("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json");
    dataset.load_images_in_parallel();

    auto controller = nrc::NeRFTrainingController(dataset);

    cudaStream_t stream;
	CUDA_ASSERT_SUCCESS(cudaStreamCreate(&stream));
    
    controller.train_step(stream);
    
    // Launch the "helloCuda" kernel on the device
    helloCuda<<<1, 1>>> ();

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    return 0;
}
