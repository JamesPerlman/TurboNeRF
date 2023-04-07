#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "../common.h"

#include "../math/transform4f.cuh"
#include "../math/tuple-math.cuh"

#include "../utils/camera-kernels.cuh"

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
	
	// equality operator
	inline __host__ bool operator==(const DistortionParams& other) const {
		return
			k1 == other.k1 &&
			k2 == other.k2 &&
			k3 == other.k3 &&
			p1 == other.p1 &&
			p2 == other.p2;
	}

	inline __host__ __device__ bool is_any_nonzero() const {
		return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || p1 != 0.0f || p2 != 0.0f;
	}
};

struct Camera {
	int2 resolution;
	float near;
	float far;
	float2 focal_length;
	float2 resolution_f;
	float2 principal_point;
	float2 shift;
	float2 offset_px;
	Transform4f transform;
	DistortionParams dist_params;
	bool show_image_planes = true;

	// constructor
	Camera(
		int2 resolution,
		float near,
		float far,
		float2 focal_length,
		float2 principal_point,
		float2 shift,
		Transform4f transform,
		DistortionParams dist_params = DistortionParams()
	)
		: resolution(resolution)
		, resolution_f{(float)resolution.x, (float)resolution.y}
		, near(near)
		, far(far)
		, focal_length(focal_length)
		, principal_point(principal_point)
		, shift(shift)
		, offset_px(shift * resolution_f - principal_point)
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
			float2{ 0.0f, 0.0f },
			Transform4f::Identity()
		) {};

	// returns a ray in the world's coordinate system
	inline __device__ Ray global_ray_at_pixel_xy(
		float x,
		float y
	) const {
		// todo: depth of field
		// http://marc-b-reynolds.github.io/math/2017/01/08/SquareDisc.html
		if (dist_params.is_any_nonzero()) {
			const float xd = (x - principal_point.x) / focal_length.x;
			const float yd = (y - principal_point.y) / focal_length.y;

			float xu, yu;
			radial_and_tangential_undistort(
				xd, yd,
				dist_params.k1, dist_params.k2, dist_params.k3,
				dist_params.p1, dist_params.p2,
				1e-9f,
				10,
				xu, yu
			);

			x = xu * focal_length.x + principal_point.x;
			y = yu * focal_length.y + principal_point.y;
		}
		
		// this represents a position at a plane 1 unit away from the camera's origin
		float3 v = {
			(x + offset_px.x) / focal_length.x,
			(y + offset_px.y) / focal_length.y,
			1.0f
		};

		// magnitude of the position vector
		float v_len = l2_norm(v);

		// transform the direction vector to global coordinates
		float3 global_dir = transform.mmul_ul3x3(v);

		// we need to normalize the global direction vector
		global_dir = global_dir / l2_norm(global_dir);

		// the ray's origin is at a plane `near` units away from the camera's origin
		float3 global_ori = transform.get_translation() + near * (v_len * global_dir);
		
		return Ray{
			global_ori, // origin, at near plane in global coordinates
			global_dir // direction, normalized
		};
	}

	// equality operator
	inline __host__ bool operator==(const Camera& other) const {
		return
			resolution == other.resolution &&
			near == other.near &&
			far == other.far &&
			focal_length == other.focal_length &&
			principal_point == other.principal_point &&
			shift == other.shift &&
			offset_px == other.offset_px &&
			transform == other.transform &&
			dist_params == other.dist_params;
	}

	// inequality operator
	inline __host__ bool operator!=(const Camera& other) const {
		return !(*this == other);
	}
};

TURBO_NAMESPACE_END
