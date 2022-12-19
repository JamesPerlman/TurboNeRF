#pragma once

#include "../common.h"

#include <vector>
#include <string>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "camera.hpp"
#include "training-image.hpp"

using namespace std;
using json = nlohmann::json;

NRC_NAMESPACE_BEGIN

struct Dataset {
	vector<Camera> cameras;
	vector<TrainingImage> images;

	static Dataset from_file(string file_path);
};

NRC_NAMESPACE_END
