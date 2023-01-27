#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "../common.h"

#include "../utils/linalg.cuh"
#include "ray.h"

NRC_NAMESPACE_BEGIN

struct Camera {
	float near;
	float far;
	float2 focal_length;
	int2 pixel_dims;
	float2 pixel_dims_f;
	float2 sensor_size;
	Matrix4f transform;

	// constructor
	Camera(float near, float far, float2 focal_length, int2 pixel_dims, float2 sensor_size, Matrix4f transform) {
		this->near = near;
		this->far = far;
		this->focal_length = focal_length;
		this->pixel_dims = pixel_dims;
		this->pixel_dims_f = make_float2(float(pixel_dims.x), float(pixel_dims.y));
		this->sensor_size = sensor_size;
		this->transform = transform;
	}

	// member functions

	// returns a ray in the camera's local coordinate system
	inline NRC_HOST_DEVICE Ray local_ray_at_pixel_xy(const uint32_t& x, const uint32_t& y) const {
		
		// sx and sy are the corresponding x and y in the sensor rect's 2D coordinate system
		// this will put rays at pixel centers
		float sx = sensor_size.x * ((float(x) + 0.5f) / (pixel_dims_f.x) - 0.5f);
		float sy = sensor_size.y * ((float(y) + 0.5f) / (pixel_dims_f.y) - 0.5f);

		float3 pix_pos = make_float3(sx, sy, 1.0f);

		float3 ray_d = pix_pos;
		float3 ray_o = near * ray_d;

		return Ray{ ray_o, ray_d };
	}
};


NRC_NAMESPACE_END
