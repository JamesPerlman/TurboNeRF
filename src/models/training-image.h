#pragma once

#include "../common.h"

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <stbi/stb_image.h>
#include <tiny-cuda-nn/gpu_memory.h>

using namespace std;

NRC_NAMESPACE_BEGIN

struct TrainingImage {
	Eigen::Vector2i dimensions;
	string filepath = "";
	std::shared_ptr<stbi_uc> data_cpu;
	int channels = 0;
	
	TrainingImage(string filepath, Eigen::Vector2i dimensions);
	
	void load_cpu(int n_channels = 0);
};

NRC_NAMESPACE_END
