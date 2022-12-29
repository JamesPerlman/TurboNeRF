#pragma once

#include "../common.h"

#include <Eigen/Dense>
#include <json/json.hpp>

NRC_NAMESPACE_BEGIN

template<typename Scalar, int Rows, int Cols>
Eigen::Array<Scalar, Rows, Cols> from_json(const nlohmann::json& data) {
    Eigen::Array<Scalar, Rows, Cols> a;
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            a(i, j) = data[i][j];
        }
    }
    return a;
}

Eigen::Matrix4f from_json(const nlohmann::json& data) {
    return from_json<float, 4, 4>(data);
}

NRC_NAMESPACE_END
