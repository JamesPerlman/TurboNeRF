#include "../common.h"

#include "training-image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stbi/stb_image.h>

using namespace std;

NRC_NAMESPACE_BEGIN

TrainingImage::TrainingImage(string filepath, Eigen::Vector2i size) {
	this->filepath = filepath;
	this->size = size;
}

void TrainingImage::load() {

	data = stbi_load(filepath.c_str(), &size.x(), &size.y(), &channels, 0);
}

NRC_NAMESPACE_END
