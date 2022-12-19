#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

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
