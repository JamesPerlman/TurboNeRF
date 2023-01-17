/**
 * AlphaTensor Matrix Multiplication (4x4) in CUDA
 * 
 * From:
 * Fawzi, A., Balog, M., Huang, A. et al. Discovering faster matrix multiplication algorithms with reinforcement learning.
 * Nature 610, 47–53 (2022). https://doi.org/10.1038/s41586-022-05172-4
 */

#include "../common.h"

#include "linalg.cuh"

NRC_NAMESPACE_BEGIN

// this is for boolean matrices or modulo-2 algebra
NRC_HOST_DEVICE Matrix4f alphatensor_mmul4x4(const Matrix4f &a, const Matrix4f &b) {
    const float& a11 = a.m00; const float& a12 = a.m01; const float& a13 = a.m02; const float& a14 = a.m03;
    const float& a21 = a.m10; const float& a22 = a.m11; const float& a23 = a.m12; const float& a24 = a.m13;
    const float& a31 = a.m20; const float& a32 = a.m21; const float& a33 = a.m22; const float& a34 = a.m23;
    const float& a41 = a.m30; const float& a42 = a.m31; const float& a43 = a.m32; const float& a44 = a.m33;

    const float& b11 = b.m00; const float& b12 = b.m01; const float& b13 = b.m02; const float& b14 = b.m03;
    const float& b21 = b.m10; const float& b22 = b.m11; const float& b23 = b.m12; const float& b24 = b.m13;
    const float& b31 = b.m20; const float& b32 = b.m21; const float& b33 = b.m22; const float& b34 = b.m23;
    const float& b41 = b.m30; const float& b42 = b.m31; const float& b43 = b.m32; const float& b44 = b.m33;

    const float h1 = a11 * b13;
    const float h2 = (a11 + a31 + a33) * (b11 + b31 + b33);
    const float h3 = (a11 + a31 + a34) * (b12 + b42 + b43);
    const float h4 = (a13 + a21 + a23) * (b13 + b14 + b34);
    const float h5 = (a11 + a31) * (b11 + b12 + b13 + b31 + b33 + b42 + b43);
    const float h6 = (a13 + a23) * (b13 + b14 + b32 + b33 + b34 + b42 + b43);
    const float h7 = (a14 + a43 + a44) * (b31 + b33 + b41);
    const float h8 = (a14 + a41 + a44) * (b13 + b14 + b44);
    const float h9 = (a13 + a23 + a24) * (b32 + b42 + b43);
    const float h10 = (a14 + a44) * (b13 + b14 + b31 + b33 + b41 + b43 + b44);
    const float h11 = a33 * (b11 + b22 + b23 + b31 + b32);
    const float h12 = (a12 + a32 + a33) * (b22 + b23 + b32);
    const float h13 = a34 * (b12 + b21 + b23 + b41 + b42);
    const float h14 = (a12 + a32) * (b21 + b22 + b23 + b32 + b41);
    const float h15 = (a12 + a32 + a34) * (b21 + b23 + b41);
    const float h16 = a21 * (b12 + b14 + b22 + b23 + b34);
    const float h17 = (a12 + a21 + a22) * (b12 + b22 + b23);
    const float h18 = (a12 + a22) * (b12 + b22 + b23 + b24 + b44);
    const float h19 = a24 * (b23 + b24 + b32 + b42 + b44);
    const float h20 = (a12 + a23 + a24 + a32 + a33) * b32;
    const float h21 = (a12 + a22 + a24) * (b23 + b24 + b44);
    const float h22 = a43 * (b23 + b24 + b31 + b34 + b41);
    const float h23 = (a11 + a13 + a14 + a23 + a24 + a31 + a34) * (b42 + b43);
    const float h24 = (a12 + a42 + a43) * (b23 + b24 + b34);
    const float h25 = (a12 + a42) * (b11 + b21 + b23 + b24 + b34);
    const float h26 = (a12 + a41 + a42) * (b11 + b21 + b23);
    const float h27 = a14 * b43;
    const float h28 = (a12 + a21 + a22 + a31 + a34) * b12;
    const float h29 = (a12 + a21 + a23 + a42 + a43) * b34;
    const float h30 = (a12 + a31 + a33 + a41 + a42) * b11;
    const float h31 = a41 * (b11 + b14 + b21 + b23 + b44);
    const float h32 = (a12 + a32 + a34 + a43 + a44) * b41;
    const float h33 = (a12 + a22 + a24 + a41 + a44) * b44;
    const float h34 = (a21 + a31 + a41) * (b11 + b12 + b14);
    const float h35 = (a12 + a21 + a22 + a32 + a33) * (b22 + b23);
    const float h36 = (a12 + a24 + a32 + a43) * (b23 + b24 + b32 + b41);
    const float h37 = (a12 + a21 + a33 + a42) * (b11 + b22 + b23 + b34);
    const float h38 = (a22 + a32 + a42) * (b21 + b22 + b24);
    const float h39 = a12 * b23;
    const float h40 = a13 * b33;
    const float h41 = (a11 + a13 + a14 + a21 + a23 + a41 + a44) * (b13 + b14);
    const float h42 = (a12 + a32 + a34 + a41 + a42) * (b21 + b23);
    const float h43 = (a24 + a34 + a44) * (b41 + b42 + b44);
    const float h44 = (a23 + a33 + a43) * (b31 + b32 + b34);
    const float h45 = (a11 + a13 + a14 + a31 + a33 + a43 + a44) * (b31 + b33);
    const float h46 = (a12 + a22 + a34 + a41) * (b12 + b21 + b23 + b44);
    const float h47 = (a12 + a22 + a24 + a42 + a43) * (b23 + b24);

    const float c11 = h15 + h26 + h2 + h30 + h32 + h39 + h40 + h42 + h45 + h7;
    const float c21 = h11 + h12 + h14 + h20 + h22 + h24 + h25 + h29 + h35 + h36 + h37 + h38 + h44 + h47;
    const float c31 = h11 + h12 + h14 + h15 + h26 + h30 + h39 + h42;
    const float c41 = h15 + h22 + h24 + h25 + h26 + h32 + h39 + h42;
    const float c12 = h12 + h17 + h20 + h23 + h27 + h28 + h35 + h39 + h3 + h9;
    const float c22 = h12 + h17 + h18 + h19 + h20 + h21 + h35 + h39;
    const float c32 = h12 + h13 + h14 + h15 + h17 + h28 + h35 + h39;
    const float c42 = h13 + h14 + h15 + h18 + h19 + h21 + h32 + h33 + h36 + h38 + h42 + h43 + h46 + h47;
    const float c13 = h1 + h27 + h39 + h40;
    const float c23 = h16 + h17 + h18 + h19 + h21 + h39 + h40 + h4 + h6 + h9;
    const float c33 = h11 + h12 + h13 + h14 + h15 + h1 + h2 + h39 + h3 + h5;
    const float c43 = h10 + h22 + h24 + h25 + h26 + h27 + h31 + h39 + h7 + h8;
    const float c14 = h1 + h21 + h24 + h29 + h33 + h39 + h41 + h47 + h4 + h8;
    const float c24 = h16 + h17 + h18 + h21 + h24 + h29 + h39 + h47;
    const float c34 = h16 + h17 + h18 + h25 + h26 + h28 + h30 + h31 + h34 + h35 + h37 + h38 + h42 + h46;
    const float c44 = h21 + h24 + h25 + h26 + h31 + h33 + h39 + h47;

    return Matrix4f{
        c11, c12, c13, c14,
        c21, c22, c23, c24,
        c31, c32, c33, c34,
        c41, c42, c43, c44
    };
}

NRC_NAMESPACE_END