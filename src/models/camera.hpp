#pragma once

#include "../common.h"

#include <Eigen/Dense>

NRC_NAMESPACE_BEGIN

struct Camera {
	float near;
	float far;
	Eigen::Vector2i dimensions;
	Eigen::Vector2f focal_length;
	Eigen::Matrix4f transform;

	// constructor
	Camera(float near, float far, Eigen::Vector2i dimensions, Eigen::Vector2f focal_length, Eigen::Matrix4f transform) {
		this->near = near;
		this->far = far;
		this->dimensions = dimensions;
		this->focal_length = focal_length;
		this->transform = transform;
	}
};

NRC_NAMESPACE_END
