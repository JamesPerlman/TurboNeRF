#include <cuda_runtime.h>

#include "../common.h"

#include "training-image.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include <stbi/stb_image.h>

using namespace nrc;

void TrainingImage::load_cpu(int n_channels) {
	data_cpu.reset(stbi_load(filepath.c_str(), &dimensions.x, &dimensions.y, &channels, n_channels));
}
