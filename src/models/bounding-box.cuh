#pragma once
#include <thrust/swap.h>
#include "../common.h"
#include "ray.h"

TURBO_NAMESPACE_BEGIN

struct BoundingBox {

    float min_x;
    float min_y;
    float min_z;

    float max_x;
    float max_y;
    float max_z;

    BoundingBox() = default;

    BoundingBox(float size)
        : min_x(-0.5f * size), min_y(-0.5f * size), min_z(-0.5f * size)
        , max_x(0.5f * size), max_y(0.5f * size), max_z(0.5f * size)
    {};

    // get nearest intersection point of the ray with the bounding box

    inline __device__ bool get_ray_t_intersections(
        const float& ori_x, const float& ori_y, const float& ori_z,
        const float& dir_x, const float& dir_y, const float& dir_z,
        const float& idir_x, const float& idir_y, const float& idir_z,
        float& tmin, float& tmax
    ) const {
        // Compute the minimum and maximum intersection points for each axis
        float t1_x = (min_x - ori_x) * idir_x;
        float t2_x = (max_x - ori_x) * idir_x;
        float t1_y = (min_y - ori_y) * idir_y;
        float t2_y = (max_y - ori_y) * idir_y;
        float t1_z = (min_z - ori_z) * idir_z;
        float t2_z = (max_z - ori_z) * idir_z;

        if (t1_x > t2_x) thrust::swap(t1_x, t2_x);
        if (t1_y > t2_y) thrust::swap(t1_y, t2_y);
        if (t1_z > t2_z) thrust::swap(t1_z, t2_z);

        // Assign tmin and tmax using the intersection points
        tmin = fmaxf(t1_x, fmaxf(t1_y, t1_z));
        tmax = fminf(t2_x, fminf(t2_y, t2_z));

        // return true if the ray intersects the bounding box
        return tmax >= tmin;
    }

    // get if the point is inside the bounding box

    inline NRC_HOST_DEVICE bool contains(const float& x, const float& y, const float& z) const {
        return x >= min_x && x <= max_x
            && y >= min_y && y <= max_y
            && z >= min_z && z <= max_z;
    }

    inline NRC_HOST_DEVICE float pos_to_unit_x(const float& x) const { return (x - min_x) / (max_x - min_x); }
    inline NRC_HOST_DEVICE float pos_to_unit_y(const float& y) const { return (y - min_y) / (max_y - min_y); }
    inline NRC_HOST_DEVICE float pos_to_unit_z(const float& z) const { return (z - min_z) / (max_z - min_z); }

    inline NRC_HOST_DEVICE float unit_to_pos_x(const float& x) const { return x * (max_x - min_x) + min_x; }
    inline NRC_HOST_DEVICE float unit_to_pos_y(const float& y) const { return y * (max_y - min_y) + min_y; }
    inline NRC_HOST_DEVICE float unit_to_pos_z(const float& z) const { return z * (max_z - min_z) + min_z; }

    inline NRC_HOST_DEVICE float size() const { return (max_x - min_x); };

    inline NRC_HOST_DEVICE float volume() const { return (max_x - min_x) * (max_y - min_y) * (max_z - min_z); };

    // equality operator
    inline NRC_HOST_DEVICE bool operator==(const BoundingBox& other) const {
        return min_x == other.min_x && min_y == other.min_y && min_z == other.min_z
            && max_x == other.max_x && max_y == other.max_y && max_z == other.max_z;
    }

    // inequality operator
    inline NRC_HOST_DEVICE bool operator!=(const BoundingBox& other) const {
        return !(*this == other);
    }
};

TURBO_NAMESPACE_END
