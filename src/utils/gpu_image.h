#pragma once

#include "../common.h"
#include <tiny-cuda-nn/common.h>

NRC_NAMESPACE_BEGIN

void save_buffer_to_image(std::string filename, tcnn::network_precision_t* data, uint32_t width, uint32_t height, uint32_t channels);
void save_buffer_to_image(std::string filename, float* data, uint32_t width, uint32_t height, uint32_t channels);

NRC_NAMESPACE_END