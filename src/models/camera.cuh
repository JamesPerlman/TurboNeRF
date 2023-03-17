#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "../common.h"

#include "../utils/linalg/transform4f.cuh"
#include "ray.h"

TURBO_NAMESPACE_BEGIN

struct DistortionParams {
	float k1;
	float k2;
	float k3;

	float p1;
	float p2;

	// constructor
	DistortionParams(float k1 = 0.0f, float k2 = 0.0f, float k3 = 0.0f, float p1 = 0.0f, float p2 = 0.0f)
		: k1(k1), k2(k2), k3(k3), p1(p1), p2(p2) { }
};

struct Camera {
	const int2 resolution;
	const float near;
	const float far;
	const float2 focal_length;
	const float2 resolution_f;
	const float2 principal_point;
	const Transform4f transform;
	const DistortionParams dist_params;

	// constructor
	Camera(
		int2 resolution,
		float near,
		float far,
		float2 focal_length,
		float2 principal_point,
		Transform4f transform,
		DistortionParams dist_params = DistortionParams()
	)
		: resolution(resolution)
		, resolution_f(make_float2(resolution.x, resolution.y))
		, near(near)
		, far(far)
		, focal_length(focal_length)
		, principal_point(principal_point)
		, transform(transform)
		, dist_params(dist_params)
	{ };

	// default constructor
	Camera()
		: Camera(
			int2{ 0, 0 },
			0.0f,
			0.0f,
			float2{ 0.0f, 0.0f },
			float2{ 0.0f, 0.0f },
			Transform4f::Identity()
		) {};

	// returns a ray in the camera's local coordinate system

	inline __device__ Ray local_ray_at_pixel_xy(
		const float& x,
		const float& y
	) const {
		const float cx = principal_point.x;
		const float cy = principal_point.y;
		
		const float xn = (x - cx) / focal_length.x;
		const float yn = (y - cy) / focal_length.y;
		
		float3 ray_o = make_float3(xn, yn, near);
		float3 ray_d = ray_o;

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

TURBO_NAMESPACE_END
