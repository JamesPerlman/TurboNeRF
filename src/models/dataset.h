#pragma once

#include <functional>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "../common.h"

#include "bounding-box.cuh"
#include "camera.cuh"
#include "training-image.cuh"

using namespace std;
using json = nlohmann::json;

NRC_NAMESPACE_BEGIN

struct Dataset {
	vector<Camera> cameras;
	vector<TrainingImage> images;
	uint32_t n_pixels_per_image;
	uint32_t n_channels_per_image;
	int2 image_dimensions;
	BoundingBox bounding_box;

	Dataset(string file_path);
	Dataset() = default;
	void load_images_in_parallel(std::function<void(const size_t, const TrainingImage&)> post_load_image = {});
};

NRC_NAMESPACE_END
