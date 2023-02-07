#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <json/json.hpp>
#include <set>

#include "common.h"
#include "main.h"
#include "controllers/nerf-training-controller.h"
#include "controllers/nerf-rendering-controller.h"
#include "core/occupancy-grid.cuh"
#include "models/camera.cuh"
#include "models/dataset.h"
#include "models/render-buffer.cuh"
#include "models/render-request.cuh"
#include "services/nerf-manager.cuh"
#include "utils/linalg/transform4f.cuh"

#include "utils/coordinate-transformations.cuh"
#include "utils/linalg/transform4f.cuh"

#include <tiny-cuda-nn/common.h>
#include "utils/nerf-constants.cuh"

#include "integrations/blender.cuh"

using namespace tcnn;
using namespace nrc;
int main()
{
	test();
	// path to downloaded dataset
	std::string DATASET_PATH = "E:\\2022\\nerf-library\\testdata\\lego\\transforms.json";
	// path to write images to
	std::string OUTPUT_PATH = "H:\\";

	cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

	nrc::Dataset dataset = nrc::Dataset(DATASET_PATH);
	// auto dataset = nrc::Dataset("E:\\2022\\nerf-library\\FascinatedByFungi2022\\big-white-chanterelle\\transforms.json");
	auto nerf_manager = nrc::NeRFManager();

	// printf("%lu", grid.max_index());
	auto nerf = nerf_manager.create_trainable_nerf(dataset.bounding_box);

	// set up training controller
	auto trainer = nrc::NeRFTrainingController(dataset, nerf, NeRFConstants::batch_size);
	trainer.prepare_for_training();

	// set up rendering controller
	auto renderer = nrc::NeRFRenderingController();
	constexpr int IMG_SIZE = 1024;
	auto render_buffer = nrc::RenderBuffer(IMG_SIZE, IMG_SIZE);

	auto camera_transform = nrc::Transform4f::Identity();
	auto cam6 = dataset.cameras[6];
	auto cam0 = dataset.cameras[6];

	// fetch nerfs as pointers
	auto proxy_ptrs = nerf_manager.get_proxies();

	for (int i = 0; i < 1024 * 10; ++i) {
		trainer.train_step();
		// every 16 training steps, update the occupancy grid

		if (i % 16 == 0 && i > 0) {
			// only threshold to 50% after 256 training steps, otherwise select 100% of the cells
			const float cell_selection_threshold = i > 256 ? 0.5f : 1.0f;
			trainer.update_occupancy_grid(cell_selection_threshold);
		}

		if (i % 16 == 0 && i > 0) {
			float progress = (float)i / (360.f * 16.0f);
			float tau = 2.0f * 3.14159f;
			auto tform = nrc::Transform4f::Rotation(progress * tau, 0.0f, 1.0f, 0.0f) * cam0.transform;
			auto render_cam = nrc::Camera(
				cam0.near,
				cam0.far,
				cam0.focal_length,
				make_int2(IMG_SIZE, IMG_SIZE),
				cam0.sensor_size,
				tform,
				cam0.dist_params
			);

			auto render_request = nrc::RenderRequest(render_cam, proxy_ptrs, &render_buffer);
			renderer.request_render(render_request);
			printf("Done!\n");
			render_request.output->save_image(OUTPUT_PATH + fmt::format("img-{}.png", i), stream);
		}
	}

	// Wait for the kernel to finish executing
	cudaDeviceSynchronize();

	cudaStreamDestroy(stream);
	return 0;
}
