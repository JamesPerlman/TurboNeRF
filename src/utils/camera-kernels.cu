#include "../math/device-math.cuh"
#include "camera-kernels.cuh"

TURBO_NAMESPACE_BEGIN

__global__ void generate_undistorted_pixel_map_kernel(
    const uint32_t n_pixels,
    const Camera camera,
    float* __restrict__ out_buf
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_pixels) {
        return;
    }
    
    const uint32_t& w = camera.resolution.x;

    const float& k1 = camera.dist_params.k1;
    const float& k2 = camera.dist_params.k2;
    const float& k3 = camera.dist_params.k3;

    const float& p1 = camera.dist_params.p1;
    const float& p2 = camera.dist_params.p2;

    const int y = divide(idx, w);
    const int x = idx - y * w;

    const float xd = (float(x) - camera.principal_point.x) / camera.focal_length.x;
    const float yd = (float(y) - camera.principal_point.y) / camera.focal_length.y;

    float xu, yu;
    radial_and_tangential_undistort(
        xd, yd,
        k1, k2, k3,
        p1, p2,
        1e-9f,
        10,
        xu, yu
    );

    out_buf[idx + 0 * n_pixels] = xu * camera.focal_length.x + camera.principal_point.x;
    out_buf[idx + 1 * n_pixels] = yu * camera.focal_length.y + camera.principal_point.y;
}

TURBO_NAMESPACE_END
