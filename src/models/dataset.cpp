#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include <cstddef>

#include "json-bindings/eigen-json.hpp"
#include "dataset.hpp"

using namespace Eigen;
using namespace std;
using namespace filesystem;
using json = nlohmann::json;
using namespace nrc;

Dataset Dataset::from_file(string file_path) {
    ifstream input_file(file_path);
    json json_data;
    input_file >> json_data;

    Dataset dataset;
    vector<Camera> cameras;
    vector<TrainingImage> images;

    Vector2i dimensions(json_data["w"], json_data["h"]);

    // TODO: per-camera focal length
    Vector2f focal_length(json_data["fl_x"], json_data["fl_y"]);

    path base_dir = path(file_path).parent_path(); // get the parent directory of file_path

    for (json frame : json_data["frames"]) {
        float near = frame["near"];
        float far = frame["far"];

        Matrix4f camera_matrix = nrc::from_json(frame["transform_matrix"]);
        // TODO: per-camera dimensions
        cameras.emplace_back(near, far, dimensions, focal_length, camera_matrix);

        // images
        string file_path = frame["file_path"];
        path absolute_path = base_dir / file_path; // construct the absolute path using base_dir

        images.emplace_back(absolute_path.string(), dimensions);
    }

    dataset.cameras = cameras;
    dataset.images = images;

    return dataset;
}

void Dataset::load_images_in_parallel() {
    const size_t num_threads = std::thread::hardware_concurrency(); // get the number of available hardware threads

    std::vector<std::thread> threads;
    std::atomic<std::size_t> index{ 0 }; // atomic variable to track the next image to be loaded
    for (size_t i = 0; i < num_threads; ++i) {
        // create a new thread to load images
        threads.emplace_back([&] {
            std::size_t local_index;
            while ((local_index = index.fetch_add(1)) < images.size()) {
                images[local_index].load();
                std::cout << local_index + 1 << '/' << images.size() << " images loaded" << '\n';
            }
        });
    }

    // wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}
