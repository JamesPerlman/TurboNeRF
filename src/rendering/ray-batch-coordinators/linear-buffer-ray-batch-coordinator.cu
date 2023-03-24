#include "../../models/camera.cuh"
#include "../../utils/device-math.cuh"
#include "../../common.h"
#include "linear-buffer-ray-batch-coordinator.cuh"

using namespace tcnn;


TURBO_NAMESPACE_BEGIN

__global__ void generate_linear_buffer_of_rays_kernel(
    const int n_rays,
    const int stride,
    const int start_idx,
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

    if (i >= n_rays) {
        return;
    }

    const int ii = i + start_idx;
    const int w = camera->resolution.x;
    const int iy = divide(ii, w);
    const int ix = ii - iy * w; 

    fill_ray_buffers(i, stride, camera, bbox, ix, iy, pos, dir, idir, t, t_max, index, alive);
}

void LinearBufferRayBatchCoordinator::generate_rays(
    const Camera* camera,
    const BoundingBox* bbox,
    RayBatch& ray_batch,
    const cudaStream_t& stream
) {
    generate_linear_buffer_of_rays_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        n_rays,
        start_idx,
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

__global__ void copy_packed_rgba_linear_buffer_kernel(
    const int n_pixels,
    const int start_idx,
    const int buffer_stride,
    const float* __restrict__ rgba_in,
    float* __restrict__ rgba_out
) {
    // i is the linear index of the ray in the hexagonal ray batch.
    int i_in = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_in >= n_pixels) {
        return;
    }

    // determine the index in the output image
    int i_out = 4 * (i_in + start_idx);

    // copy the packed rgba values

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        rgba_out[i_out] = rgba_in[i_in];
        i_out += 1;
        i_in += buffer_stride;
    }
}

void LinearBufferRayBatchCoordinator::copy_packed(
    const int& n_rays,
    const int2& output_size,
    const int& output_stride,
    float* rgba_in,
    float* rgba_out,
    const cudaStream_t& stream
) {
    const int n_pixels = this->n_rays;
    copy_packed_rgba_linear_buffer_kernel<<<n_blocks_linear(n_pixels), n_threads_linear, 0, stream>>>(
        n_pixels,
        start_idx,
        n_rays,
        rgba_in,
        rgba_out
    );
}

TURBO_NAMESPACE_END
