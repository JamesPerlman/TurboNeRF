#include "../../models/camera.cuh"
#include "../../models/ray.h"
#include "../../models/ray-batch.cuh"
#include "../../utils/device-math.cuh"
#include "../../utils/hexagon-grid.cuh"
#include "../../common.h"
#include "hexagonal-tile-ray-batch-coordinator.cuh"

using namespace tcnn;


TURBO_NAMESPACE_BEGIN

using HexagonTile = HexagonalTileRayBatchCoordinator::HexagonTile;

__global__ void generate_hexagonal_tile_of_rays_kernel(
    const int stride,
    const HexagonTile tile,
    const Camera* __restrict__ camera,
    const BoundingBox* __restrict__ bbox,
    float* __restrict__ pos,
    float* __restrict__ dir,
    float* __restrict__ idir,
    float* __restrict__ t,
    float* __restrict__ t_max,
    int* __restrict__ index,
    bool* __restrict__ alive
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= tile.n_rays) {
        return;
    }

    int ix;
    int iy;

    hex_get_pix_xy_from_buf_idx(
        i,
        tile.H,
        tile.n_rays,
        tile.fnp1_2,
        tile.cw,
        tile.fw,
        tile.fw1,
        tile.fw1_2,
        tile.fw1_sq_4,
        ix,
        iy
    );

    ix += tile.x;
    iy += tile.y;

    const Camera cam = *camera;

    if (ix < 0 || iy < 0 || ix >= cam.resolution.x || iy >= cam.resolution.y) {
        alive[i] = false;
        return;
    }

    fill_ray_buffers(i, stride, camera, bbox, ix, iy, pos, dir, idir, t, t_max, index, alive);
}

void HexagonalTileRayBatchCoordinator::generate_rays(
    const Camera* camera,
    const BoundingBox* bbox,
    RayBatch& ray_batch,
    const cudaStream_t& stream
) {
    generate_hexagonal_tile_of_rays_kernel<<<n_blocks_linear(tile.n_rays), n_threads_linear, 0, stream>>>(
        ray_batch.stride,
        tile,
        camera,
        bbox,
        ray_batch.pos,
        ray_batch.dir,
        ray_batch.idir,
        ray_batch.t,
        ray_batch.t_max,
        ray_batch.index,
        ray_batch.alive
    );
}

__global__ void copy_packed_rgba_hexagonal_tile_kernel(
    const int n_pixels,
    const int input_stride,
    const int2 output_size,
    const HexagonTile tile,
    const float* __restrict__ rgba_in,
    float* __restrict__ rgba_out
) {
    // i is the linear index of the ray in the hexagonal ray batch.
    int i_in = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_in >= n_pixels) {
        return;
    }

    // first, we convert i to the x and y coordinates of the pixel in the hexagonal tile.
    int ix, iy;
    hex_get_pix_xy_from_buf_idx(
        i_in,
        tile.H,
        tile.n_rays,
        tile.fnp1_2,
        tile.cw,
        tile.fw,
        tile.fw1,
        tile.fw1_2,
        tile.fw1_sq_4,
        ix,
        iy
    );

    // then, we convert these xy coords to the xy coords of the output pixel.
    ix += tile.x;
    iy += tile.y;

    // make sure the output pixel is in bounds
    if (ix < 0 || iy < 0 || ix >= output_size.x || iy >= output_size.y) {
        return;
    }

    // determine the index in the output image
    int i_out = 4 * (iy * output_size.x + ix);

    // copy the packed rgba values

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        rgba_out[i_out] = rgba_in[i_in];
        i_out += 1;
        i_in += input_stride;
    }
}

void HexagonalTileRayBatchCoordinator::copy_packed(
    const int& n_rays,
    const int2& output_size,
    const int& output_stride,
    float* rgba_in,
    float* rgba_out,
    const cudaStream_t& stream
) {
    const int n_hexagon_pixels = tile.n_rays;
    copy_packed_rgba_hexagonal_tile_kernel<<<n_blocks_linear(n_hexagon_pixels), n_threads_linear, 0, stream>>>(
        n_hexagon_pixels,
        n_hexagon_pixels,
        output_size,
        tile,
        rgba_in,
        rgba_out
    );
}

TURBO_NAMESPACE_END
