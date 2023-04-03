#pragma once

#include "../common.h"

#include <memory>
#include <string>
#include <stbi/stb_image.h>

using namespace std;

TURBO_NAMESPACE_BEGIN

struct TrainingImage {
	int2 dimensions;
	string file_path = "";
	std::shared_ptr<stbi_uc> data_cpu = nullptr;
	int channels = 0;
	
	TrainingImage(string file_path, int2 dimensions) : file_path(file_path), dimensions(dimensions) {};
	void load_cpu();
	void unload_cpu();
	bool is_loaded() const;
};

TURBO_NAMESPACE_END
