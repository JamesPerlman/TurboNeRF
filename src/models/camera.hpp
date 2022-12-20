#pragma once

#include "../common.h"

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
};

NRC_NAMESPACE_END
