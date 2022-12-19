#pragma once

#include <src/common.h>
#include <Eigen/Dense>

NRC_NAMESPACE_BEGIN

struct Ray {
	Eigen::Vector3f o;
	Eigen::Vector3f d;
};

NRC_NAMESPACE_END
