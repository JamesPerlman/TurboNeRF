#pragma once

#include "../common.h"
#include "ray.h"

#include <Eigen/Dense>

NRC_NAMESPACE_BEGIN

struct Camera {
	float near;
	float far;
	Eigen::Vector2f focal_length;
	Eigen::Vector2i pixel_dims;
	Eigen::Vector2f sensor_size;
	Eigen::Matrix4f transform;

	// constructor
	Camera(float near, float far, Eigen::Vector2f focal_length, Eigen::Vector2i pixel_dims, Eigen::Vector2f sensor_size, Eigen::Matrix4f transform) {
		this->near = near;
		this->far = far;
		this->focal_length = focal_length;
		this->pixel_dims = pixel_dims;
		this->sensor_size = sensor_size;
		this->transform = transform;
	}

	// member functions
	NRC_HOST_DEVICE Ray Camera::get_ray_at_pixel_xy(uint32_t x, uint32_t y);
};

NRC_NAMESPACE_END
