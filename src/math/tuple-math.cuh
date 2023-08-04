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

// float3 % float3
inline NRC_HOST_DEVICE float3 operator%(const float3& a, const float3& b)
{
    return {fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)};
}

// division float / float3
inline NRC_HOST_DEVICE float3 operator/(const float& s, const float3& v)
{
    return {s / v.x, s / v.y, s / v.z};
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

// l2 squared norm of a float3
inline NRC_HOST_DEVICE float l2_squared_norm(const float3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// l2 norm of a float3
inline NRC_HOST_DEVICE float l2_norm(const float3& v)
{
    return sqrtf(l2_squared_norm(v));
}

// returns a normalized float3
inline NRC_HOST_DEVICE float3 normalized(const float3& v)
{
    return v / l2_norm(v);
}


// dot product of two float3
inline NRC_HOST_DEVICE float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** float2 **/

// multiplication float * float2
inline NRC_HOST_DEVICE float2 operator*(const float& s, const float2& v)
{
    return {s * v.x, s * v.y};
}

// element-wise multiplication float2 * float2
inline NRC_HOST_DEVICE float2 operator*(const float2& a, const float2& b)
{
    return {a.x * b.x, a.y * b.y};
}

// element-wise multiplication float3 * float3
inline NRC_HOST_DEVICE float3 operator*(const float3& a, const float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
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
