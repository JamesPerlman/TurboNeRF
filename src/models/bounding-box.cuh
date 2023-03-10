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

    float size_x;
	float size_y;
	float size_z;
    
	BoundingBox() = default;

    BoundingBox(float size)
        : min_x(-0.5f * size), min_y(-0.5f * size), min_z(-0.5f * size)
        , max_x(0.5f * size), max_y(0.5f * size), max_z(0.5f * size)
		, size_x(size), size_y(size), size_z(size)
    {};

    inline __device__ bool get_ray_t_intersection(
        const float& ori_x, const float& ori_y, const float& ori_z,
        const float& dir_x, const float& dir_y, const float& dir_z,
        const float& idir_x, const float& idir_y, const float& idir_z,
        float& t
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

        // Update tmin and tmax using the intersection points
        float tmin = fmaxf(t1_x, fmaxf(t1_y, t1_z));
        float tmax = fminf(t2_x, fminf(t2_y, t2_z));

        // assign the t-value of the intersection point
        t = tmin;

        // return true if the ray intersects the bounding box
        return tmax >= tmin;
    }

    inline NRC_HOST_DEVICE bool contains(const float& x, const float& y, const float& z) const {
        return x >= min_x && x <= max_x
            && y >= min_y && y <= max_y
            && z >= min_z && z <= max_z;
   }


    inline NRC_HOST_DEVICE float pos_to_unit_x(const float& x) const { return (x - min_x) / size_x; }
    inline NRC_HOST_DEVICE float pos_to_unit_y(const float& y) const { return (y - min_y) / size_y; }
	inline NRC_HOST_DEVICE float pos_to_unit_z(const float& z) const { return (z - min_z) / size_z; }

    inline NRC_HOST_DEVICE float unit_to_pos_x(const float& x) const { return x * size_x + min_x; }
    inline NRC_HOST_DEVICE float unit_to_pos_y(const float& y) const { return y * size_y + min_y; }
    inline NRC_HOST_DEVICE float unit_to_pos_z(const float& z) const { return z * size_z + min_z; }
};

TURBO_NAMESPACE_END
