#pragma once

#include "../common.h"

#include <vector>
#include <string>

#include <Eigen/Dense>
#include <json/json.hpp>

#include "camera.h"
#include "training-image.h"

using namespace std;
using json = nlohmann::json;

NRC_NAMESPACE_BEGIN

struct Dataset {
	vector<Camera> cameras;
	vector<TrainingImage> images;
	uint32_t n_pixels_per_image;

	static Dataset from_file(string file_path);
	void Dataset::load_images_in_parallel();

private:
	void update_dataset_properties();
};

NRC_NAMESPACE_END
