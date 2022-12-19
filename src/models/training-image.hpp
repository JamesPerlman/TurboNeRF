#pragma once

#include "../common.h"

#include <Eigen/Dense>
#include <string>
#include <stbi/stbi_wrapper.h>

using namespace std;

NRC_NAMESPACE_BEGIN

struct TrainingImage {
	Eigen::Vector2i size;
	string filepath;
	unsigned char* data;
	int channels;

	TrainingImage(string filepath, Eigen::Vector2i size);
	void load();
};

NRC_NAMESPACE_END
