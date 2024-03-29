#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include "dataset.h"
#include "../math/transform4f.cuh"
#include "../math/tuple-math.cuh"

using namespace std;
using namespace filesystem;
using json = nlohmann::json;
using namespace turbo;

// helper function for image dimension patch
struct int2_less {
    __host__ __device__ bool operator()(const int2& a, const int2& b) const {
        return (a.x != b.x) ? a.x < b.x : a.y < b.y;
    }
};

void Dataset::load_transforms() {

    if (!file_path.has_value()) {
        throw runtime_error("No file path specified");
    }

    ifstream input_file(file_path.value());

    if (!input_file) {
        throw runtime_error("Could not open file: " + file_path->string());
    }

    json json_data;
    input_file >> json_data;

    uint32_t n_frames = json_data["frames"].size();

    cameras.reserve(n_frames);
    images.reserve(n_frames);

    float w = json_data.value("w", 0.0f);
    float h = json_data.value("h", 0.0f);

    int2 global_image_dims = make_int2((int)w, (int)h);

    uint32_t aabb_size = std::min(json_data.value("aabb_scale", 16), 128);
    float scene_scale = json_data.value("scene_scale", 1.0f);

    // principal point
    float global_cx = json_data.value("cx", 0.5f * w);
    float global_cy = json_data.value("cy", 0.5f * h);
    
    // if "fl_x" and "fl_y" are specified, these values are the focal lengths for their respective axes (in pixels)
    float global_fl_x, global_fl_y;
    if (json_data.contains("camera_angle_x")) {
        float ca_x = json_data["camera_angle_x"];
        float ca_y = json_data.value("camera_angle_y", ca_x);
        global_fl_x = 0.5f * w / tanf(0.5f * ca_x);
        global_fl_y = 0.5f * h / tanf(0.5f * ca_y);
    } else {
        global_fl_x = json_data.value("fl_x", 1000.0f);
        global_fl_y = json_data.value("fl_y", global_fl_x);
    }

    bounding_box = BoundingBox((float)aabb_size);

    float global_near = json_data.value("near", 0.05f);
    float global_far = json_data.value("far", 128.0f);

    DistortionParams global_dist_params(
        json_data.value("k1", 0.0f),
        json_data.value("k2", 0.0f),
        json_data.value("k3", 0.0f),
        json_data.value("p1", 0.0f),
        json_data.value("p2", 0.0f)
    );

    path base_dir = file_path->parent_path(); // get the parent directory of file_path

    for (json frame : json_data["frames"]) {
        
        // images
        string img_path = frame["file_path"];
        path absolute_path = base_dir / img_path; // construct the absolute path using base_dir

        if (!exists(absolute_path)) {
            continue;
        }

        int2 frame_image_dims = make_int2(frame.value("w", global_image_dims.x), frame.value("h", global_image_dims.y));

        images.emplace_back(absolute_path.string(), frame_image_dims);
        
        float near = scene_scale * frame.value("near", global_near);
        float far = scene_scale * frame.value("far", global_far);

        Transform4f transform_matrix(frame["transform_matrix"]);

        Transform4f camera_matrix = Transform4f::Scale(scene_scale) * transform_matrix.from_nerf();

        const float cx = frame.value("cx", global_cx);
        const float cy = frame.value("cy", global_cy);

        const float fl_x = frame.value("fl_x", global_fl_x);
        const float fl_y = frame.value("fl_y", global_fl_y); // potential bug, global_fl_y is not used

        DistortionParams dist_params(
            frame.value("k1", global_dist_params.k1),
            frame.value("k2", global_dist_params.k2),
            frame.value("k3", global_dist_params.k3),
            frame.value("p1", global_dist_params.p1),
            frame.value("p2", global_dist_params.p2)
        );

        // TODO: per-camera dimensions
        cameras.emplace_back(
            frame_image_dims,
            near,
            far,
            float2{fl_x, fl_y},
            float2{cx, cy},
            float2{0.0f, 0.0f},
            camera_matrix,
            dist_params
        );
    }

    if (images.empty()) {
        throw runtime_error("No valid images in this dataset!");
    };

    // TODO: support multiple image dimensions

    // count number of images for each dimension
    std::map<int2, int, int2_less> img_dims_and_counts;
    for (const auto& img : images) {
        if (img_dims_and_counts.find(img.dimensions) == img_dims_and_counts.end()) {
            img_dims_and_counts[img.dimensions] = 0;
        }
        img_dims_and_counts[img.dimensions]++;
    }

    // if there are multiple image dimensions, we need to keep only the images with the most common dimensions
    if (img_dims_and_counts.size() > 1) {
        contains_multiple_image_dims = true;
        int2 most_common_dimensions;
        int most_common_dimensions_count = 0;
        for (const auto& [dims, count] : img_dims_and_counts) {
            if (count > most_common_dimensions_count) {
                most_common_dimensions_count = count;
                most_common_dimensions = dims;
            }
        }

        // filter out all images with dimensions that are not the most common
        images.erase(
            std::remove_if(
                images.begin(),
                images.end(),
                [&most_common_dimensions](const auto& img) {
                    return img.dimensions != most_common_dimensions;
                }
            ),
            images.end()
        );

        cameras.erase(
            std::remove_if(
                cameras.begin(),
                cameras.end(),
                [&most_common_dimensions](const auto& cam) {
                    return cam.resolution != most_common_dimensions;
                }
            ),
            cameras.end()
        );
    }

    image_dimensions = images[0].dimensions;
    n_pixels_per_image = image_dimensions.x * image_dimensions.y;

    // remove excess allocated images
    images.shrink_to_fit();
    cameras.shrink_to_fit();
}

Dataset::Dataset(
    const BoundingBox& bounding_box,
    const vector<Camera>& cameras,
    const vector<TrainingImage>& images
)   : bounding_box(bounding_box)
    , cameras(cameras)
    , images(images)
{
    image_dimensions = images[0].dimensions;
    n_pixels_per_image = image_dimensions.x * image_dimensions.y;
}

// this method was written (mostly) by ChatGPT!
void Dataset::load_images(Dataset::ImageLoadCallback post_load_image) {
    const int num_threads = std::thread::hardware_concurrency(); // get the number of available hardware threads
    const int n_images_total = images.size();
    std::vector<std::thread> threads;
    std::atomic<int> index{ 0 }; // atomic variable to track the next image to be loaded
    for (int i = 0; i < num_threads; ++i) {
        // create a new thread to load images
        threads.emplace_back([&] {
            int local_index;
            while ((local_index = index.fetch_add(1)) < n_images_total) {
                images[local_index].load_cpu();
                if (post_load_image) {
                    post_load_image(images[local_index], local_index, n_images_total);
                }
            }
        });
    }

    // wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

void Dataset::unload_images() {
    for (auto& image : images) {
        image.unload_cpu();
    }
}

bool Dataset::is_loaded_cpu() const {
    for (const auto& image : images) {
        if (!image.is_loaded()) {
            return false;
        }
    }
    return true;
}

json Dataset::to_json() const {
    json json_data;

    json_data["w"] = image_dimensions.x;
    json_data["h"] = image_dimensions.y;

    json_data["aabb_scale"] = bounding_box.size();
    json_data["scene_scale"] = 1.0f;

    json frames = json::array();

    for (size_t i = 0; i < cameras.size(); ++i) {
        json frame;

        frame["near"] = cameras[i].near;
        frame["far"] = cameras[i].far;

        frame["cx"] = cameras[i].principal_point.x;
        frame["cy"] = cameras[i].principal_point.y;

        frame["fl_x"] = cameras[i].focal_length.x;
        frame["fl_y"] = cameras[i].focal_length.y;

        frame["k1"] = cameras[i].dist_params.k1;
        frame["k2"] = cameras[i].dist_params.k2;
        frame["k3"] = cameras[i].dist_params.k3;
        frame["p1"] = cameras[i].dist_params.p1;
        frame["p2"] = cameras[i].dist_params.p2;

        frame["transform_matrix"] = cameras[i].transform.to_nerf().to_matrix().to_json();

        path image_path(images[i].file_path);
        path relative_path = image_path.lexically_relative(file_path.value().parent_path());
        
        // convert to posix
        string path_string = relative_path.string();
        std::replace(path_string.begin(), path_string.end(), '\\', '/');

        // save as posix
        frame["file_path"] = path_string;

        frames.push_back(frame);
    }

    json_data["frames"] = frames;

    return json_data;
}

Dataset Dataset::copy() const {
    Dataset dataset(bounding_box, cameras, images);
    dataset.file_path = file_path;
    return dataset;
}
