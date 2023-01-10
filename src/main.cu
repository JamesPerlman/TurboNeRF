#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <json/json.hpp>

#include "common.h"
#include "main.h"
#include "models/camera.cuh"
#include "models/dataset.h"
#include "models/cascaded-occupancy-grid.cuh"
#include "models/render-buffer.cuh"
#include "models/render-request.cuh"
#include "controllers/nerf-training-controller.h"
#include "controllers/nerf-rendering-controller.h"
#include "services/nerf-manager.cuh"
#include "utils/linalg.cuh"

int main()
{
	nrc::Dataset dataset = nrc::Dataset("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json");

    auto nerf_manager = nrc::NeRFManager();
    
    // nrc::OccupancyGrid grid(1);

    // printf("%lu", grid.max_index());
    
    cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

    auto nerf = nerf_manager.create_trainable_nerf(stream, dataset.bounding_box.size_x);

    // set up training controller
    auto trainer = nrc::NeRFTrainingController(dataset, nerf);
    trainer.prepare_for_training(stream, 2<<19);

    // set up rendering controller
    auto renderer = nrc::NeRFRenderingController();
    float* rgba;

    CUDA_CHECK_THROW(cudaMallocManaged(&rgba, 1024 * 1024 * 4 * sizeof(float)));
    auto render_buffer = nrc::RenderBuffer(1024, 1024, rgba);

    auto camera_transform = nrc::Matrix4f::Identity();

    auto render_cam = nrc::Camera(0.0f, 8.0f, make_float2(100.0f, 100.0f), make_int2(1024, 1024), make_float2(1.0f, 1.0f), camera_transform);

    // fetch nerfs as pointers
    std::vector<const nrc::NeRF*> nerf_ptrs;
    for (const auto& nerf : nerf_manager.get_nerfs()) {
        nerf_ptrs.push_back(&nerf);
    }

    auto render_request = nrc::RenderRequest(render_buffer, render_cam, nerf_ptrs);

    for (int i = 0; i < 1000; ++i) {
        trainer.train_step(stream);

        if (i % 100 == 0) {
            renderer.request_render(stream, render_request);
        }
    }

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();
    return 0;
}
