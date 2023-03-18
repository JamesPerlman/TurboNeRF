#include "../../models/camera.cuh"
#include "../../models/ray.h"
#include "../../models/ray-batch.cuh"
#include "../../utils/device-math.cuh"
#include "../../utils/hexagon-grid.cuh"
#include "../../common.h"
#include "hexagonal-grid-ray-batch-coordinator.cuh"

using namespace tcnn;


TURBO_NAMESPACE_BEGIN

__global__ void generate_hexagonal_grid_of_rays_kernel(
    const int n_rays,
    const int stride,
    const int grid_width,
    const int2 grid_offset,
    const int W,
    const int H,
    const int cw,
    const Camera* __restrict__ camera,
    const BoundingBox* __restrict__ bbox,
    float* __restrict__ pos,
    float* __restrict__ dir,
    float* __restrict__ idir,
    float* __restrict__ t,
    int* __restrict__ index,
    bool* __restrict__ alive
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) {
        return;
    }
    const int gj = divide(i, grid_width);
    const int gi = i - grid_width * gj;

    int ix;
    int iy;

    hex_get_xy_from_ij(gi, gj, H, W, cw, ix, iy);

    ix = ix - grid_offset.x + W / 2;
    iy = iy - grid_offset.y + H / 2;

    const Camera cam = *camera;

    const Ray local_ray = cam.local_ray_at_pixel_xy(ix, iy);
    const Ray global_ray = cam.global_ray_from_local_ray(local_ray);

    fill_ray_buffers(i, stride, global_ray, bbox, pos, dir, idir, t, index, alive);
}

void HexagonalGridRayBatchCoordinator::generate_rays(
    const Camera* camera,
    const BoundingBox* bbox,
    RayBatch& ray_batch,
    const cudaStream_t& stream
) {
    const int n_rays = grid_dims.x * grid_dims.y;
    generate_hexagonal_grid_of_rays_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        n_rays,
        grid_dims.x,
        grid_offset,
        W,
        H,
        cw,
        camera,
        bbox,
        ray_batch.pos,
        ray_batch.dir,
        ray_batch.idir,
        ray_batch.t,
        ray_batch.index,
        ray_batch.alive
    );
}

__global__ void copy_packed_rgba_hexagonal_grid_kernel(
    const int n_pixels,
    const int stride,
    const int output_width,
    const int grid_width,
    const int grid_height,
    const int2 grid_offset,
    const int W,
    const int H,
    const int cw,
    const float* __restrict__ rgba_in,
    float* __restrict__ rgba_out
) {
    // i is the index of the pixel in the output image
    int i_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i_out >= n_pixels) {
        return;
    }

    // first we must get the output x,y indices
    int y = divide(i_out, output_width); // (i_out / output_width)
    int x = i_out - output_width * y;    // (i_out % output_width)

    x += grid_offset.x;
    y += grid_offset.y;

    // then we get the grid i,j indices
    int i, j;
    hex_get_ij_from_xy(x, y, H, W, cw, i, j);

    // now we can get the index of the pixel in the input buffer
    int i_in = j * grid_width + i;
    i_out = 4 * i_out;

    // finally we copy the packed pixel values
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        rgba_out[i_out] = rgba_in[i_in];
        i_out += 1;
        i_in += stride;
    }
}


void HexagonalGridRayBatchCoordinator::copy_packed(
    const int& n_rays,
    const int2& output_size,
    const int& output_stride,
    float* rgba_in,
    float* rgba_out,
    const cudaStream_t& stream
) {
    const int n_output_pixels = output_size.x * output_size.y;

    copy_packed_rgba_hexagonal_grid_kernel<<<n_blocks_linear(n_output_pixels), n_threads_linear, 0, stream>>>(
        n_output_pixels,
        grid_dims.x * grid_dims.y,
        output_size.x,
        grid_dims.x,
        grid_dims.y,
        grid_offset,
        W,
        H,
        cw,
        rgba_in,
        rgba_out
    );
}

TURBO_NAMESPACE_END
