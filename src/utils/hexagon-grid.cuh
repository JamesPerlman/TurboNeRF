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

    float fx = (buf_idx / 2);
    float a = fx - fw + fw1_sq_4;
    float y = sqrtf(a) - fw1_2;
    return 1 + (int)y;
}

/**
 * Each two-row rectangle starts at some memory offset.  This function returns that offset.
 * 
 */

inline __device__ int buf_offset_from_hex_rect_idx(
    const float& rect_idx,
    const float& fw,
    const float& fw1_2, // 0.5 * fw1
    const float& fw1_sq_4 // 0.25 * fw1 * fw1
) {
    float x = rect_idx - 1;
    float a = x + fw1_2;
    int half_offset = a * a + fw - fw1_sq_4;

    return 2 * half_offset;
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

inline __device__ int n_pix_total_in_hex(
    const int& h,
    const int& fw,
    const int& fw1_2,
    const int& fw1_sq_4
) {
    const int rect_idx = h / 4;
    return 2 * buf_offset_from_hex_rect_idx(rect_idx, fw, fw1_2, fw1_sq_4);
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
 * Pixels are indexed by scanning the hexagon row-by-row, 0 to x+, 0 to y+.
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

inline __device__ void hex_pix_xy_from_buf_idx(
    const int& buf_idx,
    const int& cw,
    const int& H,
    int& x,
    int& y
) {
    const float fw = cw;
    const float fw1 = (fw + 1);
    const float fw1_2 = 0.5f * fw1;
    const float fw1_sq_4 = 0.25f * fw1 * fw1;

    const int n_pix = n_pix_total_in_hex(H, fw, fw1_2, fw1_sq_4);
    const float fnp1_2 = ((float)n_pix - 1.0f) / 2.0f;

    // adjusted buffer index (avoids some branching)
    const float idx = fnp1_2 - fabsf((float)buf_idx - fnp1_2);

    // calculate necessary properties
    const int rect_idx = hex_rect_idx_from_buf_idx(buf_idx, fw, fw1, fw1_2, fw1_sq_4);
    const int n_rect_pix = n_pix_for_hex_rect_at_idx(rect_idx, cw);
    const int rect_offset = buf_offset_from_hex_rect_idx(rect_idx, fw, fw1_2, fw1_sq_4);

    const int n_pix_per_row = n_rect_pix / 2;
    
    int idx_in_row = idx - rect_offset;

    // both branches share this value as a base
    int _y = rect_idx + (int)(idx_in_row >= n_pix_per_row);

    // tiny bit of branch divergence, could be a fun to-do to eliminate it :)
    if (buf_idx < n_pix / 2) {
        _y = _y + rect_idx;
    } else {
        idx_in_row = n_rect_pix - idx_in_row - 1;
        _y = H - _y - 1;
    }

    const int _x = hex_row_x_offset(_y, H) + idx_in_row % n_pix_per_row;

    x = _x;
    y = _y;
}
