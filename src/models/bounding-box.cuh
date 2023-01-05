#pragma once

#include "../common.h"
#include "ray.h"

NRC_NAMESPACE_BEGIN

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
    
    /*
    inline NRC_HOST_DEVICE bool BoundingBox::intersects_ray(const Ray& ray) {
        float tmin, tmax;

        // Initialize tmin and tmax with the ray's parameter range
        tmin = 0.0f;
        tmax = ray.d.maxCoeff();

        // Compute the inverse direction of the ray
        float inv_dir_x = 1.0f / ray.d.x();
        float inv_dir_y = 1.0f / ray.d.y();
        float inv_dir_z = 1.0f / ray.d.z();
        
        // Compute the minimum and maximum intersection points for each axis
        float t1_x = (bbox.min.x() - ray.o.x()) * inv_dir_x;
        float t2_x = (bbox.max.x() - ray.o.x()) * inv_dir_x;
        float t1_y = (bbox.min.y() - ray.o.y()) * inv_dir_y;
        float t2_y = (bbox.max.y() - ray.o.y()) * inv_dir_y;
        float t1_z = (bbox.min.z() - ray.o.z()) * inv_dir_z;
        float t2_z = (bbox.max.z() - ray.o.z()) * inv_dir_z;

        // Make sure t1 is smaller than t2 for each axis
        if (t1_x > t2_x) std::swap(t1_x, t2_x);
        if (t1_y > t2_y) std::swap(t1_y, t2_y);
        if (t1_z > t2_z) std::swap(t1_z, t2_z);

        // Update tmin and tmax using the intersection points
        tmin = std::max(tmin, std::max(t1_x, std::max(t1_y, t1_z)));
        tmax = std::min(tmax, std::min(t2_x, std::min(t2_y, t2_z)));

        // If tmin is larger than tmax, the ray does not intersect the bounding box
        return tmin <= tmax;
    }
    */

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

NRC_NAMESPACE_END
