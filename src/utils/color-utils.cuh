#pragma once

#include "../common.h"

// Thank you https://entropymine.com/imageworsener/srgbformula/ and ChatGPT

NRC_NAMESPACE_BEGIN

// fast math version
inline __device__ float __srgb_to_linear(const float& srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
	} else {
        return __powf(__fdividef((srgb + 0.055f), 1.055f), 2.4f);
	}
}

inline __device__ float srgb_to_linear(const float& srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    } else {
        return powf((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

// fast math version
inline __device__ float __linear_to_srgb(const float& linear) {
	if (linear <= 0.0031308f) {
		return linear * 12.92f;
	} else {
		return 1.055f * __powf(linear, __fdividef(1.0f, 2.4f)) - 0.055f;
	}
}

inline __device__ float linear_to_srgb(const float& linear) {
    if (linear <= 0.0031308f) {
        return linear * 12.92f;
    } else {
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }
}

// float to byte
inline __device__ uint32_t rgba32f_to_rgba8(const float& r, const float& g, const float& b, const float& a) {
    return (uint32_t(__saturatef(a) * 255.0f) << 24) |
           (uint32_t(__saturatef(b) * 255.0f) << 16) |
           (uint32_t(__saturatef(g) * 255.0f) << 8) |
           (uint32_t(__saturatef(r) * 255.0f));
}

NRC_NAMESPACE_END
