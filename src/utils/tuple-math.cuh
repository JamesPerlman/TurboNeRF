#pragma once
#include "../common.h"

TURBO_NAMESPACE_BEGIN

/** float3 **/

// multiplication float * float3
inline NRC_HOST_DEVICE float3 operator*(const float& s, const float3& v)
{
    return {s * v.x, s * v.y, s * v.z};
}

// division float3 / float
inline NRC_HOST_DEVICE float3 operator/(const float3& v, const float& s)
{
    return {v.x / s, v.y / s, v.z / s};
}

// addition float3 + float3
inline NRC_HOST_DEVICE float3 operator+(const float3& a, const float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// subtraction float3 - float3
inline NRC_HOST_DEVICE float3 operator-(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

// equality float3 == float3
inline NRC_HOST_DEVICE bool operator==(const float3& a, const float3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// inequality float3 != float3
inline NRC_HOST_DEVICE bool operator!=(const float3& a, const float3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

/** float2 **/

// element-wise multiplication float2 * float2
inline NRC_HOST_DEVICE float2 operator*(const float2& a, const float2& b)
{
    return {a.x * b.x, a.y * b.y};
}

// element-wise subtraction float2 - float2
inline NRC_HOST_DEVICE float2 operator-(const float2& a, const float2& b)
{
    return {a.x - b.x, a.y - b.y};
}

// equality float2 == float2
inline NRC_HOST_DEVICE bool operator==(const float2& a, const float2& b)
{
    return a.x == b.x && a.y == b.y;
}

// inequality float2 != float2
inline NRC_HOST_DEVICE bool operator!=(const float2& a, const float2& b)
{
    return a.x != b.x || a.y != b.y;
}

/** int2 **/

// equality int2 == int2
inline NRC_HOST_DEVICE bool operator==(const int2& a, const int2& b)
{
    return a.x == b.x && a.y == b.y;
}

// inequality int2 != int2
inline NRC_HOST_DEVICE bool operator!=(const int2& a, const int2& b)
{
    return a.x != b.x || a.y != b.y;
}

TURBO_NAMESPACE_END
