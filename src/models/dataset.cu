#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include "dataset.h"
#include "../utils/linalg/transform4f.cuh"

using namespace std;
using namespace filesystem;
using json = nlohmann::json;
using namespace turbo;

Dataset::Dataset(const string& file_path) {
    ifstream input_file(file_path);
    json json_data;
    input_file >> json_data;

	uint32_t n_frames = json_data["frames"].size();

    cameras.reserve(n_frames);
    images.reserve(n_frames);

    float w = json_data.value("w", 0.0f);
    float h = json_data.value("h", 0.0f);

    image_dimensions = make_int2((int)w, (int)h);
    n_pixels_per_image = (uint32_t)(w * h);
    n_channels_per_image = 4;

    float cx = json_data.value("cx", 0.5f * w);
    float cy = json_data.value("cy", 0.5f * h);
    float2 principal_point = make_float2(cx, cy);
    
    // TODO: per-camera focal length
    // if "fl_x" and "fl_y" are specified, these values are the focal lengths for their respective axes (in pixels)
    float fl_x, fl_y;
    if (json_data.contains("fl_x")) {
        fl_x = json_data["fl_x"];
        fl_y = json_data.value("fl_y", fl_x);
    } else if (json_data.contains("camera_angle_x")) {
        float ca_x = json_data["camera_angle_x"];
        float ca_y = json_data.value("camera_angle_y", ca_x);
        fl_x = 0.5f * w / tanf(0.5f * ca_x);
        fl_y = 0.5f * h / tanf(0.5f * ca_y);
    }
    
    float2 focal_length = make_float2(fl_x, fl_y);

    uint32_t aabb_size = std::min(json_data.value("aabb_scale", 16), 128);
    bounding_box = BoundingBox((float)aabb_size);

    float global_near = json_data.value("near", 0.05f);
    float global_far = json_data.value("far", 128.0f);

    DistortionParams dist_params(
        json_data.value("k1", 0.0f),
        json_data.value("k2", 0.0f),
        json_data.value("k3", 0.0f),
        json_data.value("p1", 0.0f),
        json_data.value("p2", 0.0f)
    );

    path base_dir = path(file_path).parent_path(); // get the parent directory of file_path

    for (json frame : json_data["frames"]) {
        float near = frame.value("near", global_near);
        float far = frame.value("far", global_far);

        Transform4f transform_matrix(frame["transform_matrix"]);

        Transform4f camera_matrix = transform_matrix.from_nerf();

        // TODO: per-camera dimensions
        cameras.emplace_back(
            image_dimensions,
            near,
            far,
            focal_length,
            principal_point,
            camera_matrix,
            dist_params
        );

        // images
        string file_path = frame["file_path"];
        path absolute_path = base_dir / file_path; // construct the absolute path using base_dir

        // only add the image if it exists
        if (exists(absolute_path)) {
            images.emplace_back(absolute_path.string(), image_dimensions);
        }
    }

    // remove excess allocated images
    images.shrink_to_fit();
}

// this method was written (mostly) by ChatGPT!
void Dataset::load_images_in_parallel(std::function<void(const size_t, const TrainingImage&)> post_load_image) {
    const size_t num_threads = std::thread::hardware_concurrency(); // get the number of available hardware threads

    std::vector<std::thread> threads;
    std::atomic<std::size_t> index{ 0 }; // atomic variable to track the next image to be loaded
    for (size_t i = 0; i < num_threads; ++i) {
        // create a new thread to load images
        threads.emplace_back([&] {
            std::size_t local_index;
            while ((local_index = index.fetch_add(1)) < images.size()) {
                images[local_index].load_cpu(n_channels_per_image);
                if (post_load_image) {
                    post_load_image(local_index, images[local_index]);
                }
            }
        });
    }

    // wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}
