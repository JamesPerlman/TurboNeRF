#include <cuda_runtime.h>

#include "../common.h"

#include "training-image.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include <stbi/stb_image.h>

using namespace turbo;

void TrainingImage::load_cpu() {
	data_cpu.reset(stbi_load(file_path.c_str(), &dimensions.x, &dimensions.y, &channels, 4));
}
