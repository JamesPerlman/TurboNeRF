#pragma once

#include "../common.h"

#include <memory>
#include <string>
#include <stbi/stb_image.h>

using namespace std;

TURBO_NAMESPACE_BEGIN

struct TrainingImage {
	int2 dimensions;
	string filepath = "";
	std::shared_ptr<stbi_uc> data_cpu;
	int channels = 0;
	
	TrainingImage(string filepath, int2 dimensions) : filepath(filepath), dimensions(dimensions) {};
	void load_cpu(int n_channels = 0);
};

TURBO_NAMESPACE_END
