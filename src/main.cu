#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <json/json.hpp>

#include "common.h"
#include "main.h"
#include "models/dataset.h"
#include "models/cascaded-occupancy-grid.cuh"
#include "controllers/nerf-training-controller.h"

// Declare a global device function
__global__ void helloCuda(void)
{
    printf("Hello World from GPU!!\n");
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
}

int main()
{
	nrc::Dataset dataset = nrc::Dataset("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json");
    
    // nrc::OccupancyGrid grid(1);

    // printf("%lu", grid.max_index());
    auto controller = nrc::NeRFTrainingController(dataset);
    
    cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

    controller.prepare_for_training(stream, 2<<21);
    
    for (int i = 0; i < 1000; ++i) {
        controller.train_step(stream);
    }
    
    // Launch the "helloCuda" kernel on the device
    helloCuda<<<1, 1>>> ();

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    return 0;
}
