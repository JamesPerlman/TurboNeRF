/**
 * 4x4 transformation matrix helpers, optimized for 3D graphics
 * by James Perlman, 2023 (with lots of help from ChatGPT and GitHub Copilot!) 
 */

#include "../../common.h"
#include "../linalg.cuh"

NRC_NAMESPACE_BEGIN

struct Transform4f
{
    float m00, m01, m02, m03;
    float m10, m11, m12, m13;
    float m20, m21, m22, m23;

    Transform4f() = default;

    Transform4f(Matrix4f m)
        : m00(m.m00), m01(m.m01), m02(m.m02), m03(m.m03)
        , m10(m.m10), m11(m.m11), m12(m.m12), m13(m.m13)
        , m20(m.m20), m21(m.m21), m22(m.m22), m23(m.m23)
    {};

    Transform4f(
        const float& m00, const float& m01, const float& m02, const float& m03,
        const float& m10, const float& m11, const float& m12, const float& m13,
        const float& m20, const float& m21, const float& m22, const float& m23)
        : m00(m00), m01(m01), m02(m02), m03(m03)
        , m10(m10), m11(m11), m12(m12), m13(m13)
        , m20(m20), m21(m21), m22(m22), m23(m23)
    {};

    void print() const
    {
        printf("%f %f %f %f\n", m00, m01, m02, m03);
        printf("%f %f %f %f\n", m10, m11, m12, m13);
        printf("%f %f %f %f\n", m20, m21, m22, m23);
        printf("%f %f %f %f\n", 0.0f, 0.0f, 0.0f, 1.0f);
        printf("\n");
    }

    float determinant() const
    {
        return 0.0f
            + m00 * (m11 * m22 - m12 * m21)
            - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20);
    }

    Transform4f inverse() const
    {
        const float m11_x_m22_m_m12_x_m21 = m11 * m22 - m12 * m21;
        const float m10_x_m22_m_m12_x_m20 = m10 * m22 - m12 * m20;
        const float m10_x_m21_m_m11_x_m20 = m10 * m21 - m11 * m20;

        const float det = m00 * (m11_x_m22_m_m12_x_m21) - m01 * (m10_x_m22_m_m12_x_m20) + m02 * (m10_x_m21_m_m11_x_m20);

        const float i_det = 1.0f / det;

        const float m12_x_m23_m_m13_x_m22 = m12 * m23 - m13 * m22;
        const float m11_x_m23_m_m13_x_m21 = m11 * m23 - m13 * m21;
        const float m10_x_m23_m_m13_x_m20 = m10 * m23 - m13 * m20;

        return Transform4f{
            +(m11_x_m22_m_m12_x_m21) * i_det,
            -(m01 * m22 - m02 * m21) * i_det,
            +(m01 * m12 - m02 * m11) * i_det,
            -(m01 * (m12_x_m23_m_m13_x_m22) - m02 * (m11_x_m23_m_m13_x_m21) + m03 * (m11_x_m22_m_m12_x_m21)) * i_det,
            -(m10_x_m22_m_m12_x_m20) * i_det,
            +(m00 * m22 - m02 * m20) * i_det,
            -(m00 * m12 - m02 * m10) * i_det,
            +(m00 * (m12_x_m23_m_m13_x_m22) - m02 * (m10_x_m23_m_m13_x_m20) + m03 * (m10_x_m22_m_m12_x_m20)) * i_det,
            +(m10_x_m21_m_m11_x_m20) * i_det,
            -(m00 * m21 - m01 * m20) * i_det,
            +(m00 * m11 - m01 * m10) * i_det,
            -(m00 * (m11_x_m23_m_m13_x_m21) - m01 * (m10_x_m23_m_m13_x_m20) + m03 * (m10_x_m21_m_m11_x_m20)) * i_det,
        };
    }
};

NRC_NAMESPACE_END
