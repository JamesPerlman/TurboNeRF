#include "../common.h"

#include "training-image.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stbi/stb_image.h>

using namespace std;

NRC_NAMESPACE_BEGIN

TrainingImage::TrainingImage(string filepath, Eigen::Vector2i dimensions) {
	this->filepath = filepath;
	this->dimensions = dimensions;
}

void TrainingImage::load() {
	data = stbi_load(filepath.c_str(), &dimensions.x(), &dimensions.y(), &channels, 0);
}

NRC_NAMESPACE_END
