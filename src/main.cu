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

	auto nerf = nerf_manager.create_trainable_nerf(stream, dataset.bounding_box);

	// set up training controller
	auto trainer = nrc::NeRFTrainingController(dataset, nerf);
	trainer.prepare_for_training(stream, 1<<20);

	// set up rendering controller
	auto renderer = nrc::NeRFRenderingController();
	float* rgba;

	CUDA_CHECK_THROW(cudaMallocManaged(&rgba, 1024 * 1024 * 4 * sizeof(float)));
	auto render_buffer = nrc::RenderBuffer(1024, 1024, rgba);

	auto camera_transform = nrc::Matrix4f::Identity();
	auto train_cam = dataset.cameras[6];
	auto render_cam = nrc::Camera(
		train_cam.near,
		train_cam.far,
		train_cam.focal_length,
		make_int2(1024, 1024),
		train_cam.sensor_size,
		train_cam.transform
	);
	// fetch nerfs as pointers
	std::vector<nrc::NeRF*> nerf_ptrs;
	for (auto& nerf : nerf_manager.get_nerfs()) {
		nerf_ptrs.emplace_back(nerf);
	}

	auto render_request = nrc::RenderRequest(render_buffer, render_cam, nerf_ptrs);

	for (int i = 0; i < 300; ++i) {
		trainer.train_step(stream);

		if (i % 1 == 0 && i > 0) {
			render_request.output.clear(stream);
			renderer.request_render(stream, render_request);
			render_request.output.save_image(stream, fmt::format("H:\\test-render-{}.png", i));
		}
	}

	// Wait for the kernel to finish executing
	cudaDeviceSynchronize();
	return 0;
}
