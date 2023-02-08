#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <json/json.hpp>
#include <set>
#include <unordered_map>

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



// Put strings in in .bss part of program
static const std::string DEFAULT_DATASET_PATH = "E:\\2022\\nerf-library\\testdata\\lego\\transforms.json";
static const std::string DEFAULT_OUTPUT_PATH = "H:\\";
static const std::string HELP_OUTPUT = R"(
	Usage: NeRFRenderCore -i <path_to_json> -o <path_to_output_dir>

	Options:
	-h,			Show this help message
	-i,			Path to trainings data, e.g.: E:\\nerf\\transforms.json (windows) or /home/developer/transforms.json
	-o,			Path to directory with rendering output, e.g.: E:\\ (windows) or /home/developer/ 

)";

bool validPathDirectory(const std::string&); 
bool isJSONFile(const std::string &);

int main(int argc, char* argv[])
{
	std::unordered_map<std::string, std::string> args;
	// If we don't have any arguments, we proceed with the default input and output locations.
	// We start with i = 1 because i = 0 is the program name.
	for(int i = 1; i < argc; ++i) { 
		const std::string arg = argv[i];
		if (arg == "-h") {
			std::cout << HELP_OUTPUT;
			return 0;
		} else if (arg.size() == 2 && i + 1 < argc) {
			args[arg] = argv[++i];
		} else if (arg.size() == 2) {
			std::cout << fmt::format("Invalid argument {} {}", args[arg], argv[++i]);
			return -1;
		} else {
			std::cout << fmt::format("Invalid argument {}", args[arg]);
			return -1;
		}
	}


	if (!args["-h"].empty()) {
		std::cout << HELP_OUTPUT;
		return 0;
	}

	const std::string DATASET_PATH = args["-i"].empty() ? DEFAULT_DATASET_PATH : args["-i"];
	const std::string OUTPUT_PATH =  args["-o"].empty() ? DEFAULT_OUTPUT_PATH : args["-o"];

	std::cout << DATASET_PATH << std::endl;
	std::cout << OUTPUT_PATH << std::endl;
	
	if (!validPathDirectory(OUTPUT_PATH) || !isJSONFile(DATASET_PATH)) {
		return -1;
	}
	test();

	cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

	nrc::Dataset dataset = nrc::Dataset(DATASET_PATH);
	// auto dataset = nrc::Dataset("E:\\2022\\nerf-library\\FascinatedByFungi2022\\big-white-chanterelle\\transforms.json");
	auto nerf_manager = nrc::NeRFManager();

	// printf("%lu", grid.max_index());
	auto nerf = nerf_manager.create_trainable(dataset.bounding_box);

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
				cam0.resolution,
				cam0.near,
				cam0.far,
				cam0.focal_length,
				cam0.view_angle,
				tform,
				cam0.dist_params
			);

			auto render_request = nrc::RenderRequest(render_cam, proxy_ptrs, &render_buffer);
			renderer.request(render_request);
			printf("Done!\n");
			render_request.output->save_image(OUTPUT_PATH + fmt::format("img-{}.png", i), stream);
		}
	}

	// Wait for the kernel to finish executing
	cudaDeviceSynchronize();

	cudaStreamDestroy(stream);
	return 0;
}

bool validPathDirectory(const std::string &pathToDir) {
	std::filesystem::path path(pathToDir);
	if (!path.has_relative_path() || path.filename() != "") {
		std::cout << fmt::format("Invalid Directory - Should end either with \\\\ on windows or / on linux", pathToDir);
		return false;
	}
	if (std::filesystem::is_directory(path) && path.has_relative_path()) {
		return true;
	} else {
		std::cout << fmt::format("Directory invalid: {}", pathToDir);
		return false;
	}
}

bool isJSONFile(const std::string &pathToFile) {
	
	if (!std::filesystem::exists(pathToFile)) {
		std::cout << fmt::format("{} doesn't exist", pathToFile);
		return false;
	}
	
	if (std::filesystem::path(pathToFile).extension() == ".json")	{
		return true;
	} else {
		std::cout << fmt::format("Fileformat not JSON : {}", pathToFile);
		return false;
	}
}