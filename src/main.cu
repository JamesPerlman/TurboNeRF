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

#include "utils/alphatensor_mmul4x4.cuh"
#include "utils/coordinate-transformations.cuh"
#include "utils/linalg/transform4f.cuh"

#include <tiny-cuda-nn/common.h>

#include "models/cascaded-occupancy-grid.cuh"
int main()
{

	float ray_ori_x = 1.0f;
	float ray_ori_y = 1.0f;
	float ray_ori_z = 1.0f;
	float ray_dir_x = 1.0f / sqrt(3.0f);
	float ray_dir_y = 1.0f / sqrt(3.0f);
	float ray_dir_z = 1.0f / sqrt(3.0f);
	float ray_idir_x = 1.0f / ray_dir_x;
	float ray_idir_y = 1.0f / ray_dir_y;
	float ray_idir_z = 1.0f / ray_dir_z;

	nrc::CascadedOccupancyGrid grid(5);

	const float dt = grid.get_dt_to_next_voxel(
		ray_ori_x, ray_ori_y, ray_ori_z,
		ray_dir_x, ray_dir_y, ray_dir_z,
		ray_idir_x, ray_idir_y, ray_idir_z,
		0.01 * sqrt(3.0f) / 1024.0f,
		4
	);

	nrc::Dataset dataset = nrc::Dataset("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json");
	// auto dataset = nrc::Dataset("E:\\2022\\nerf-library\\FascinatedByFungi2022\\big-white-chanterelle\\transforms.json");
	auto nerf_manager = nrc::NeRFManager();
	
	// nrc::OccupancyGrid grid(1);

	// printf("%lu", grid.max_index());
	
	cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

	auto nerf = nerf_manager.create_trainable_nerf(stream, dataset.bounding_box);

	// set up training controller
	auto trainer = nrc::NeRFTrainingController(dataset, nerf);
	trainer.prepare_for_training(stream, 1<<21);

	// set up rendering controller
	auto renderer = nrc::NeRFRenderingController();
	float* rgba;

	CUDA_CHECK_THROW(cudaMallocManaged(&rgba, 1024 * 1024 * 4 * sizeof(float)));
	auto render_buffer = nrc::RenderBuffer(1024, 1024, rgba);

	auto camera_transform = nrc::Matrix4f::Identity();
	auto cam6 = dataset.cameras[6];
	auto cam0 = dataset.cameras[6];

	// fetch nerfs as pointers
	std::vector<nrc::NeRF*> nerf_ptrs;
	for (auto& nerf : nerf_manager.get_nerfs()) {
		nerf_ptrs.emplace_back(nerf);
	}

	for (int i = 0; i <= 100000; ++i) {
		trainer.train_step(stream);

		// every 16 training steps, update the occupancy grid

		if (i % 16 == 0 && i > 0) {
			// only threshold to 50% after 256 training steps, otherwise select 100% of the cells
			const float cell_selection_threshold = i > 256 ? 0.5f : 1.0f;
			trainer.update_occupancy_grid(stream, cell_selection_threshold);
		}

		if (i % 1000 == 0 && i > 0) {
			float progress = 0.0f;//(float)i / (30.0f * 60.0f);
			float tau = 2.0f * 3.14159f;
			auto tform = nrc::Matrix4f::Rotation(3.0f * progress * tau, 0.0f, 1.0f, 0.0f) * cam0.transform;
			auto render_cam = nrc::Camera(
				cam0.near,
				cam0.far,
				cam0.focal_length,
				make_int2(1024, 1024),
				cam0.sensor_size,
				tform
			);

			auto render_request = nrc::RenderRequest(render_buffer, render_cam, nerf_ptrs);
			render_request.output.clear(stream);
			renderer.request_render(stream, render_request);
			render_request.output.save_image(stream, fmt::format("H:\\test-render-2\\step-{}.png", i));
		}
	}

	// Wait for the kernel to finish executing
	cudaDeviceSynchronize();
	return 0;
}
