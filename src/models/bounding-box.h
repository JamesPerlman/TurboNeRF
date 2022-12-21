#pragma once

#include <Eigen/Dense>

#include "../common.h"
#include "ray.h"

NRC_NAMESPACE_BEGIN

struct BoundingBox {
	Eigen::Vector3f min;
	Eigen::Vector3f max;
	
	BoundingBox(Eigen::Vector3f min, Eigen::Vector3f max) {
		this->min = min;
		this->max = max;
	}

	NRC_HOST_DEVICE bool intersects_ray(const Ray& ray);
};

NRC_NAMESPACE_END
