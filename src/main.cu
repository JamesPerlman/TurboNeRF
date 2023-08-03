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
#include "models/camera.cuh"
#include "models/dataset.h"
#include "models/render-request.cuh"
#include "services/nerf-manager.cuh"
#include "services/runtime-manager.cuh"
#include "render-targets/cuda-render-buffer.cuh"
#include "math/transform4f.cuh"

#include "math/transform4f.cuh"

#include <tiny-cuda-nn/common.h>
#include "utils/nerf-constants.cuh"

#include "integrations/blender.cuh"

using namespace tcnn;
using namespace turbo;



// Put strings in in .bss part of program
static const std::string DEFAULT_DATASET_PATH = "E:\\2022\\nerf-library\\testdata\\lego\\transforms.json";
static const std::string DEFAULT_OUTPUT_PATH = "H:\\";
static const std::string HELP_OUTPUT = R"(
    Usage: TurboNeRF -i <path_to_json> -o <path_to_output_dir>

    Options:
    -h,         Show this help message
    -i,         Path to training data, e.g.: E:\\nerf\\transforms.json (windows) or /home/developer/transforms.json
    -o,         Path to render output, e.g.: E:\\ (windows) or /home/developer/ 

)";

bool check_is_directory(const std::string&); 
bool check_is_json_file(const std::string&);

int main(int argc, char* argv[])
{
    auto runtime_manager = turbo::RuntimeManager();
    bool can_run_turbo_nerf = runtime_manager.check_runtime();
    if (!can_run_turbo_nerf) {
        return -1;
    }
    
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
    
    if (!check_is_directory(OUTPUT_PATH) || !check_is_json_file(DATASET_PATH)) {
        return -1;
    }

    cudaStream_t stream;
    CUDA_CHECK_THROW(cudaStreamCreate(&stream));

    turbo::Dataset dataset = turbo::Dataset(DATASET_PATH);
    dataset.load_transforms();
    
    auto nerf_manager = std::make_shared<NeRFManager>();


    
    // set up rendering controller
    auto renderer = turbo::NeRFRenderingController(RenderPattern::LinearChunks);
    int IMG_SIZE = dataset.cameras[0].resolution.x;
    auto render_buffer = turbo::CUDARenderBuffer();
    render_buffer.set_size(IMG_SIZE, IMG_SIZE);

    auto camera_transform = turbo::Transform4f::Identity();
    auto cam6 = dataset.cameras[6];
    auto cam0 = dataset.cameras[6];

    // auto loaded_id = nerf_manager.load("H:\\dozer.turbo");
    // std::vector<NeRFProxy*> nerfs(1);
    // nerfs[0] = nerf_manager.get_proxy_ptr(loaded_id);
    // auto render_request = std::make_shared<RenderRequest>(
    //  cam6,
    //  nerfs,
    //  &render_buffer,
    //  RenderModifiers(),
    //  RenderFlags::Final,
    //  // on_complete
    //  []() {},
    //  // on_progress, save image
    //  [&](float progress) {
    //      // render_buffer.save_image(OUTPUT_PATH + fmt::format("img-{}-{}.png", progress, i), stream);
    //  }
    // );

    // renderer.submit(render_request);
    // render_buffer.save_image("H:\\saved_render.png", stream);

    // return 0;

    turbo::NeRFProxy* proxy = nerf_manager->create();
    proxy->attach_dataset(dataset);

    // set up training controller
    auto trainer = turbo::NeRFTrainingController(proxy);
    trainer.setup_data();
    trainer.load_images([](int a, int b) {
        printf("Loading images: %d / %d\n", a, b);
    });

    // fetch nerfs as pointers
    turbo::NeRFProxy* proxy2 = nerf_manager->clone(proxy);
    proxy2->transform = turbo::Transform4f::Translation(0.0f, 0.0f, 0.0f);
    proxy->transform = turbo::Transform4f::Translation(0.0f, 0.5f, 0.0f);
    std::vector<NeRFProxy*> nerf_proxies{ proxy, proxy2 };

    std::vector<NeRFRenderable> renderables;
    renderables.reserve(nerf_proxies.size());
    for (auto& p : nerf_proxies) {
        renderables.emplace_back(p);
    }

    for (int i = 0; i < 3360; ++i) {
        const auto& info = trainer.train_step();
        printf("step: %d, loss: %f\n", info.step, info.loss);
        // every 16 training steps, update the occupancy grid

        if (i % 16 == 0 && i > 16) {
            // only threshold to 50% after 256 training steps, otherwise select 100% of the cells
            trainer.update_occupancy_grid(i);
        }

        if (i % 128 == 0 && i > 0) {
            float progress = (float)i / (360.f * 0.5f);
            float tau = 2.0f * 3.14159f;
            auto tform = turbo::Transform4f::Rotation(progress * tau, 0.0f, 1.0f, 0.0f) * cam0.transform;
            auto render_cam = turbo::Camera(
                make_int2(IMG_SIZE, IMG_SIZE),
                0.03f,
                cam0.far,
                cam0.focal_length,
                cam0.principal_point,
                cam0.shift,
                cam0.transform,
                cam0.dist_params
            );
            auto mods = RenderModifiers();
            auto render_request = std::make_shared<RenderRequest>(
                render_cam,
                renderables,
                &render_buffer,
                mods,
                RenderFlags::Final,
                // on_complete
                []() {},
                // on_progress, save image
                [&](float progress) {
                    // render_buffer.save_image(OUTPUT_PATH + fmt::format("img-{}-{}.png", progress, i), stream);
                }
            );

            printf("Rendering image %d\n", i);
            renderer.submit(render_request);
            printf("Saving image %d\n", i);
            render_buffer.save_image(OUTPUT_PATH + fmt::format("img-{}.png", i), stream);
            printf("Done!\n");
        }
    }

    FileManager::save(proxy, "H:\\dozer.turbo");

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    cudaStreamDestroy(stream);
    render_buffer.free();
    return 0;
}

bool check_is_directory(const std::string &path_str) {
    std::filesystem::path path(path_str);
    if (std::filesystem::is_directory(path)) {
        return true;
    } else {
        std::cout << fmt::format("Directory invalid: {}", path_str);
        return false;
    }
}

bool check_is_json_file(const std::string &path_str) {
    
    if (!std::filesystem::exists(path_str)) {
        std::cout << fmt::format("{} doesn't exist", path_str);
        return false;
    }
    
    if (std::filesystem::path(path_str).extension() == ".json") {
        return true;
    } else {
        std::cout << fmt::format("Fileformat not JSON : {}", path_str);
        return false;
    }
}