#include <cuda_runtime.h>

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

void TrainingImage::load_cpu(int n_channels) {
	data_cpu = std::shared_ptr<stbi_uc>(stbi_load(filepath.c_str(), &dimensions.x(), &dimensions.y(), &channels, n_channels));
}


NRC_NAMESPACE_END
