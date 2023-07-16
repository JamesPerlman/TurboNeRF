#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "../common.h"

#include "bounding-box.cuh"
#include "camera.cuh"
#include "training-image.cuh"

TURBO_NAMESPACE_BEGIN

struct Dataset {
    using ImageLoadCallback = std::function<void(const TrainingImage&, int, int)>;

    std::vector<Camera> cameras;
    std::vector<TrainingImage> images;
    uint32_t n_pixels_per_image;
    int2 image_dimensions;
    BoundingBox bounding_box;
    std::optional<filesystem::path> file_path;

    Dataset(const std::string& file_path) : file_path(file_path) {};
    Dataset(const BoundingBox& bounding_box, const std::vector<Camera>& cameras, const std::vector<TrainingImage>& images);
    Dataset() = default;

    void load_transforms();

    void load_images(ImageLoadCallback post_load_image = {});
    void unload_images();
    
    bool is_loaded_cpu() const;

    nlohmann::json to_json() const;
    Dataset copy() const;
};

TURBO_NAMESPACE_END
