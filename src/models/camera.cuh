#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "../common.h"

#include "../utils/linalg.cuh"
#include "ray.h"

NRC_NAMESPACE_BEGIN

struct DistortionParams {
	float k1;
	float k2;
	float k3;
	float k4;

	float p1;
	float p2;

	// constructor
	DistortionParams(float k1 = 0.0f, float k2 = 0.0f, float k3 = 0.0f, float k4 = 0.0f, float p1 = 0.0f, float p2 = 0.0f)
		: k1(k1), k2(k2), k3(k3), k4(k4), p1(p1), p2(p2) { }
};

struct Camera {
	float near;
	float far;
	float2 focal_length;
	int2 pixel_dims;
	float2 pixel_dims_f;
	float2 sensor_size;
	Matrix4f transform;
	DistortionParams distortion;

	// constructor
	Camera(float near, float far, float2 focal_length, int2 pixel_dims, float2 sensor_size, Matrix4f transform, DistortionParams distortion) {
		this->near = near;
		this->far = far;
		this->focal_length = focal_length;
		this->pixel_dims = pixel_dims;
		this->pixel_dims_f = make_float2(float(pixel_dims.x), float(pixel_dims.y));
		this->sensor_size = sensor_size;
		this->transform = transform;
		this->distortion = distortion;
	}

	// member functions

	// returns a ray in the camera's local coordinate system

	inline __device__ Ray local_ray_at_pixel_xy_index(const uint32_t& x, const uint32_t& y) const {
		
		// sx and sy are the corresponding x and y in the sensor rect's 2D coordinate system
		// this will put rays at pixel centers
		const float sx = sensor_size.x * ((float(x) + 0.5f) / (pixel_dims_f.x) - 0.5f);
		const float sy = sensor_size.y * ((float(y) + 0.5f) / (pixel_dims_f.y) - 0.5f);

		float3 pix_pos = make_float3(sx, sy, near);

		float3 ray_d = pix_pos;
		float3 ray_o = pix_pos;

		return Ray{ ray_o, ray_d };
	}
	
	inline __device__ Ray local_ray_at_pixel_xy_normalized(const float& x, const float& y) const {
		const float sx = sensor_size.x * x;
		const float sy = sensor_size.y * y;

		float3 pix_pos = make_float3(sx, sy, near);

		float3 ray_d = pix_pos;
		float3 ray_o = pix_pos;

		return Ray{ ray_o, ray_d };
	}
};


NRC_NAMESPACE_END
