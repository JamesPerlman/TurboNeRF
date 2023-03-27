#pragma once

#include "../common.h"
#include "tuple-math.cuh"
#include "transform4f.cuh"

TURBO_NAMESPACE_BEGIN

// Thanks GPT-4!
inline __device__
bool ray_plane_intersection(
    const float3& ray_ori,
    const float3& ray_dir,
    const float3& plane_center,
    const float3& plane_normal,
    const float2& plane_size,
    const Transform4f& inv_transform,
    float2& uv,
    float& t
) {
    // Calculate the intersection point between the ray and the plane.
    float denom = dot(plane_normal, ray_dir);

    // Check if the ray is parallel to the plane.
    if (abs(denom) < 1e-6)
        return false;

    t = dot(plane_center - ray_ori, plane_normal) / denom;

    // Check if intersection is behind the ray origin.
    if (t < 0)
        return false;

    float3 intersection_point = ray_ori + t * ray_dir;

    // Transform the intersection point back to the plane's local coordinate system.
    float3 local_point = inv_transform * intersection_point;

    // Check if the intersection point lies within the plane's width (w) and height (h).
    float half_width = plane_size.x / 2.0f;
    float half_height = plane_size.y / 2.0f;

    if (fabsf(local_point.x) <= half_width && fabsf(local_point.y) <= half_height) {
        uv = float2{
            (local_point.x + half_width) / plane_size.x,
            (local_point.y + half_height) / plane_size.y
        };

        return true;
    }

    return false;
};


TURBO_NAMESPACE_END
