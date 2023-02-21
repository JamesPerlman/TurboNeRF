#pragma once
#include "../common.h"
#include "device-math.cuh"

NRC_NAMESPACE_BEGIN

/**
 * This is a pseudo-regular hexagon.  The slope of the sides is 0.5 instead of 1/sqrt(3),
 * therefore it is not a perfect regular hexagon. But it looks close enough and has the
 * properties we need to pack the pixels covered by the hexagon into a linear buffer,
 * addressing them by a single index to get (x,y) pixel coordinates.
 * 
 * These hexagons can tile the plane such that every pixel is covered by exactly one hexagon.
 * 
 * 
 * W = total width
 * cw = width of central rectangular portion (recommended to be multiple of 2)
 * H = height (must be multiple of 4)
 * 
 * It is recommended to set W = 1.1 * H.  A true hexagon has W/H = 2/sqrt(3) =~ 1.1547
 * Using W = 1.1 * H gives us a "close enough" approximation with a very usable ratio,
 * and it looks very aesthetic.
 * 
 * 
 *                  0.5 * (W - cw)  
 *                  ___
 *                 |   |
 *       __________
 *      /          \
 *     /            \
 *    /              \
 *    \              /
 *     \            /
 *      \__________/
 *          cw
 * 
 * Here are some scratchpads where the following math was derived:
 * https://www.desmos.com/calculator/i8ixakbd7f
 * https://www.desmos.com/calculator/bvoqmkivjc
 */

/**
 * This just gets the width of the central rectangular portion of the hexagon.
 * 
 */
inline __host__ void hex_get_W_and_cw(
    const int& H, // height (should be a multiple of 4)
    const float& a, // aspect ratio (recommended to be 1.1)
    int& W,
    int& cw
) {
    W = (int)(a * (float)H);
    cw = W - H / 2 + 2;
}

/**
 * This gets the height of the hexagon given a requested number of pixels and an aspect ratio.
 * 
 */

inline __host__ int hex_height_for_npix_and_aspect(const float& n, const float& a) {
    const float b = 4.0f * a - 1.0f;
    return (int)(2.0f * (sqrtf(n * b) - 1.0f) / b);
}

/**
 * Due to the angle of the slope chosen, our hexagon can be broken into dual-row rectangular segments.
 * This function gets the index of the rectangle, given the index of the pixel in the buffer.
 */

inline __device__ int hex_rect_idx_from_buf_idx(
    const int& buf_idx,
    const float& fw, // cw as float
    const float& fw1, // fw + 1
    const float& fw1_2, // 0.5 * fw1
    const float& fw1_sq_4 // 0.25 * fw1 * fw1
) {

    float fx = (float)buf_idx / 2.0f;
    float a = fx - fw + fw1_sq_4;
    float y = 1.0f + sqrtf(a) - fw1_2;
    return (int)y;
}

/**
 * Each two-row rectangle starts at some memory offset.  This function returns that offset.
 * 
 */

inline __device__ __host__ int buf_offset_from_hex_rect_idx(
    const float& rect_idx,
    const float& fw,
    const float& fw1_2, // 0.5 * fw1
    const float& fw1_sq_4 // 0.25 * fw1 * fw1
) {
    float x = rect_idx - 1.0f;
    float a = x + fw1_2;
    float offset = 2.0f * (a * a + fw - fw1_sq_4);

    return (int)offset;
}

/**
 * This function returns the number of pixels per rectangular section.
 * 
 */

inline __device__ int n_pix_for_hex_rect_at_idx(const int& rect_idx, const int& cw) {
    return 2 * (2 * rect_idx + cw);
}

/**
 * This function returns the number of total pixels in a hexagon of the given dimensions.
 * 
 */

inline __host__ int n_pix_total_in_hex(
    const int& h,
    const int& c
) {
    return h * (h / 4 + c - 1);
}


/**
 * Each row is offset by some amount in the x-dimension.  This function returns that offset.
 * 
 */

inline __device__ int hex_row_x_offset(const int& y, const int& H) {
    const float fy = y;
    const float fH = H;
    return (int)(0.5f * fabsf(fy - 0.5f * (fH - 1.0f)));
}

/**
 * This function assigns the x and y coordinates of the pixel at the given index in the buffer.
 * Pixels are indexed by scanning the hexagon row-by-row, 0 to W in x+, 0 to H in y+.
 * Imagine the hexagon is aligned such that the left corner is at x=0, and the bottom edge is at y=0
 * The hexagon exists in the first quadrant.  The assigned x and y coordinates are relative to the origin:
 * 
 *  y
 *  |
 *  | __
 *  |/  \
 *  |\__/____ x 
 * 
 */

inline __device__ void hex_get_pix_xy_from_buf_idx(
    const int& buf_idx,
    const int& H,
    const int& n_pix,
    const float& fnp1_2, // ((float)n_pix - 1.0f) / 2.0f
    const int& cw,
    const float& fw,
    const float& fw1,
    const float& fw1_2,
    const float& fw1_sq_4,
    int& x,
    int& y
) {

    // adjusted buffer index (avoids some branching)
    const float idx = fnp1_2 - fabsf((float)buf_idx - fnp1_2);

    // calculate necessary properties
    const int rect_idx = hex_rect_idx_from_buf_idx(idx, fw, fw1, fw1_2, fw1_sq_4);
    const int n_rect_pix = n_pix_for_hex_rect_at_idx(rect_idx, cw);
    const int rect_offset = buf_offset_from_hex_rect_idx(rect_idx, fw, fw1_2, fw1_sq_4);

    const int n_pix_per_row = n_rect_pix / 2;

    int idx_in_rect = idx - rect_offset;

    // both branches share this value as a base
    int _y = 2 * rect_idx + (int)(idx_in_rect >= n_pix_per_row);

    // tiny bit of branch divergence, could be a fun to-do to eliminate it :)
    if (buf_idx >= n_pix / 2) {
        idx_in_rect = n_rect_pix - idx_in_rect - 1;
        _y = H - _y - 1;
    }

    const int _x = hex_row_x_offset(_y, H) + idx_in_rect % n_pix_per_row;

    x = _x;
    y = _y;
}

/**
 * Grid position getter -> given (i,j) in hexagonal grid coordinate, return (x,y) in cartesian coordinates.
 * 
 * If the aspect ratio 1.1 is used, it's recommended to keep the width at a multiple of 20.
 * This way, the hexagon will always be centered on the pixel grid, and never off by a subpixel amount.
 * 
 */

inline __device__ void hex_get_xy_from_ij(
    const int& i,
    const int& j,
    const int& H,
    const int& W,
    const int& cw,
    const float& a,
    int& x,
    int& y
) {
    x = i * (W - cw) / 2;

    // if i is odd, shift y up by half a hexagon
    // doing this without branching, otherwise we will have (threadIdx % 2) divergence (which is bad I hear...)
    int k = (int)(i & 1) * (H / 2);
    y = j * H + k;
}

/**
 * Grid position getter: given (x,y) in cartesian coordinates, assign (i,j) in hexagonal grid coordinate.
 * Scratchpad: https://www.shadertoy.com/view/dtBSDt
 * 
 */

inline __device__ void hex_get_ij_from_xy(
    const int& x,
    const int& y,
    const int& H,
    const int& W,
    const int& cw,
    int& i,
    int& j
) {
    
    // subtile width and height
    const int tw = (W + cw) / 2;
    const int th = H / 2;

    // tile x and y indices
    const int ti = divide(x, tw);
    const int tj = divide(y, th);

    // x and y-offset relative to start of tile
    const int tu = x - tw * ti;
    const int tv = y - th * tj;

    // y-offset is below the lower-left slope of hexagon in this tile
    // the -1 here aligns these hexagons to the pixel, the same way the hexagon buffers are aligned above
    const bool al = tu < ((th - tv - 1) / 2);

    // y-offset is below the upper-left slope of hexagon in this tile
    const bool bl = tu < (tv / 2);

    // tile x isOdd and y isOdd
    const bool io = ti & 1;
    const bool jo = tj & 1;

    // hexagon x-index is even
    bool xeven = jo
        ? (  (io & al) | !(io | bl) )
        : ( !(io | al) |  (io & bl) );
    
    // width of two tiles (column period)
    const int rw = 2 * tw;

    // height of two tiles is just the height of hexagon    
    const int rh = H;

    // branch divergence here, hope this isn't too bad
    if (xeven) {
        i = 2 * divide(x, rw);
        j = divide(y, rh);
    } else {
        i = 2 * divide(x + tw, rw) - 1;
        j = divide(y + th, rh);
    }
}

NRC_NAMESPACE_END
