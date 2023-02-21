#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "../common.h"

#include "../utils/linalg/transform4f.cuh"
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
	const int2 resolution;
	const float near;
	const float far;
	const float2 focal_length;
	const float2 resolution_f;
	const float2 view_angle;
	const float2 sensor_size;
	const Transform4f transform;
	const DistortionParams dist_params;

	// constructor
	Camera(
		int2 resolution,
		float near,
		float far,
		float2 focal_length,
		float2 view_angle,
		Transform4f transform,
		DistortionParams dist_params = DistortionParams()
	)
		: resolution(resolution)
		, resolution_f(float2{float(resolution.x), float(resolution.y)})
		, near(near)
		, far(far)
		, focal_length(focal_length)
		, view_angle(view_angle)
		, sensor_size(float2{2.0f * near * tanf(view_angle.x * 0.5f), 2.0f * near * tanf(view_angle.y * 0.5f)})
		, transform(transform)
		, dist_params(dist_params)
	{ };

	// returns a ray in the camera's local coordinate system

	inline __device__ Ray local_ray_at_pixel_xy_index(
		const uint32_t& x,
		const uint32_t& y
	) const {
		// sx and sy are the corresponding x and y in the sensor rect's 2D coordinate system
		// this will put rays at pixel centers
		const float sx = sensor_size.x * ((float(x) + 0.5f) / (resolution_f.x) - 0.5f);
		const float sy = sensor_size.y * ((float(y) + 0.5f) / (resolution_f.y) - 0.5f);

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

	inline __device__ Ray global_ray_from_local_ray(const Ray& local_ray) const {
		float3 global_origin = transform * local_ray.o;
		float3 global_direction = transform.mmul_ul3x3(local_ray.d);

		// normalize ray directions
		const float n = rnorm3df(global_direction.x, global_direction.y, global_direction.z);
		return Ray{ global_origin, n * global_direction };
	}
};

NRC_NAMESPACE_END
