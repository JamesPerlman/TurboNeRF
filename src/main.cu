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
#include "services/nerf-manager.cuh"

int main()
{
	nrc::Dataset dataset = nrc::Dataset("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json");

    auto nerf_manager = nrc::NeRFManager();
    
    // nrc::OccupancyGrid grid(1);

    // printf("%lu", grid.max_index());
    
    cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

    auto nerf = nerf_manager.create_trainable_nerf(stream, dataset.bounding_box.size_x);
    auto controller = nrc::NeRFTrainingController(dataset, nerf);
    controller.prepare_for_training(stream, 2<<21);
    
    for (int i = 0; i < 1000; ++i) {
        controller.train_step(stream);
    }

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();
    return 0;
}
