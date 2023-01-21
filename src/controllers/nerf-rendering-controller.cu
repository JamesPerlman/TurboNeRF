#include <tiny-cuda-nn/common.h>
#include "nerf-rendering-controller.h"
#include "../models/camera.cuh"
#include "../utils/cu-compactor.cuh"
#include "../utils/nerf-constants.cuh"
#include "../utils/parallel-utils.cuh"
#include "../utils/rendering-kernels.cuh"

using namespace nrc;
using namespace tcnn;

NeRFRenderingController::NeRFRenderingController(
    uint32_t batch_size
) {
    if (batch_size == 0) {
        // TODO: determine batch size from GPU specs
        this->batch_size = 1<<21;
    } else {
        this->batch_size = batch_size;
    }
}

void NeRFRenderingController::request_render(
    const cudaStream_t& stream,
    const RenderRequest& request
) {
    // TODO: this should happen for all NeRFs
    NeRF* nerf = request.nerfs[0];

    // TODO: enlarge workspace only on batch size or output size change
    workspace.enlarge(
        stream,
        request.output.width,
        request.output.height,
        batch_size,
        nerf->network.get_concat_buffer_width(),
        nerf->network.get_padded_output_width(),
        n_threads_linear
    );

    printf("Rendering...\n");

    // workspace.camera = request.camera
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.camera,
            &request.camera,
            sizeof(Camera),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    // workspace.bounding_box = nerf->bounding_box
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.bounding_box,
            &nerf->bounding_box,
            sizeof(BoundingBox),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    // workspace.occupancy_grid = nerf->occupancy_grid
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.occupancy_grid,
            &nerf->occupancy_grid,
            sizeof(CascadedOccupancyGrid),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    // calculate the number of pixels we need to fill
    uint32_t n_pixels = request.output.width * request.output.height;
    uint32_t n_rays_total = n_pixels;

    // double buffer indices
    int active_buf_idx = 0;
    int compact_buf_idx = 1;

    // loop over all pixels, chunked by batch size
    uint32_t n_pixels_filled = 0;
    while (n_pixels_filled < n_pixels) {
        // TODO:
        // for (auto& n in nerfs) { ... 
        // calculate the number of pixels to fill in this batch
        uint32_t n_pixels_to_fill = std::min(
            batch_size,
            n_pixels - n_pixels_filled
        );

        uint32_t n_rays = n_pixels_to_fill;

        // calculate the pixel indices to fill in this batch
        uint32_t pixel_start = n_pixels_filled;

        // generate rays for the pixels in this batch
        generate_rays_pinhole_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
            n_rays,
            batch_size,
            workspace.bounding_box,
            workspace.camera,
            workspace.ray_origin[active_buf_idx],
            workspace.ray_dir[active_buf_idx],
            workspace.ray_idir[active_buf_idx],
            workspace.ray_t[active_buf_idx],
            workspace.ray_idx[active_buf_idx],
            workspace.ray_alive,
            pixel_start
        );

        // initialize other ray properties

        // ray_sigma = 0
        CUDA_CHECK_THROW(cudaMemsetAsync(workspace.ray_sigma[active_buf_idx], 0, batch_size * sizeof(float), stream));

        // ray_active = true
        CUDA_CHECK_THROW(cudaMemsetAsync(workspace.ray_active[active_buf_idx], true, batch_size * sizeof(bool), stream));

        
        float dt_min = NeRFConstants::min_step_size;
        float dt_max = nerf->bounding_box.size_x * dt_min;
        const float cone_angle = NeRFConstants::cone_angle;

        // ray marching loop
        uint32_t n_rays_alive = n_rays;

        // TODO: march rays to bounding box first
        while (n_rays_alive > 0) {
            uint32_t network_batch = tcnn::next_multiple(n_rays_alive, tcnn::batch_size_granularity);

            // march each ray one step
            // TODO: should we march potentially multiple steps to maximize occupancy?
            march_rays_and_generate_network_inputs_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
                n_rays_alive,
                batch_size,
                network_batch,
                workspace.occupancy_grid,
                workspace.bounding_box,
                1.0f / nerf->bounding_box.size_x,
                dt_min,
                dt_max,
                cone_angle,
                workspace.ray_origin[active_buf_idx],
                workspace.ray_dir[active_buf_idx],
                workspace.ray_idir[active_buf_idx],
                workspace.ray_alive,
                workspace.ray_active[active_buf_idx],
                workspace.ray_t[active_buf_idx],
                workspace.network_pos,
                workspace.network_dir,
                workspace.network_dt
            );

            // query the NeRF network for the samples
            nerf->network.inference(
                stream,
                network_batch,
                workspace.network_pos,
                workspace.network_dir,
                workspace.network_concat,
                workspace.network_output
            );

            // CHECK_DATA(color_cpu, float, workspace.network_color, network_batch * 3);

            // check how many color values in color_cpu are >1
            // uint32_t color_values_great_than_1 = 0;

            // for (uint32_t i = 0; i < network_batch * 3; i++) {
            //     if (color_cpu[i] > 1.0f) {
            //         color_values_great_than_1++;
            //     }
            // }

            // printf("# of color values >1: %d\n", color_values_great_than_1);

            // accumulate these samples into the pixel colors
            composite_samples_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
                n_rays_alive,
                network_batch,
                request.output.stride,
                workspace.network_output,
                workspace.network_dt,
                workspace.ray_idx[active_buf_idx],
                workspace.ray_active[active_buf_idx],
                workspace.ray_alive,
                workspace.ray_sigma[active_buf_idx],
                request.output.rgba
            );

            // update how many rays are still alive
            const int n_rays_to_compact = calculate_block_counts_and_offsets(
                stream,
                n_rays_alive,
                workspace.c_block_size,
                workspace.ray_alive,
                workspace.c_block_counts,
                workspace.c_block_offsets
            );

            // if no rays are alive, we can skip compositing
            if (n_rays_to_compact == 0) {
                break;
            }
            
            // check if compaction is required
            if (n_rays_to_compact < n_rays_alive) {
                // get compacted ray indices
                generate_compaction_indices(
                    stream,
                    n_rays_alive,
                    workspace.c_block_size,
                    workspace.ray_alive,
                    workspace.c_block_offsets,
                    workspace.compact_idx
                );

                // compact ray properties via the indices
                compact_rays_kernel<<<n_blocks_linear(n_rays_to_compact), n_threads_linear, 0, stream>>>(
                    n_rays_to_compact,
                    batch_size,
                    workspace.compact_idx,

                    // input
                    workspace.ray_idx[active_buf_idx],
                    workspace.ray_active[active_buf_idx],
                    workspace.ray_t[active_buf_idx],
                    workspace.ray_origin[active_buf_idx],
                    workspace.ray_dir[active_buf_idx],
                    workspace.ray_idir[active_buf_idx],
                    workspace.ray_sigma[active_buf_idx],

                    // output
                    workspace.ray_idx[compact_buf_idx],
                    workspace.ray_active[compact_buf_idx],
                    workspace.ray_t[compact_buf_idx],
                    workspace.ray_origin[compact_buf_idx],
                    workspace.ray_dir[compact_buf_idx],
                    workspace.ray_idir[compact_buf_idx],
                    workspace.ray_sigma[compact_buf_idx]
                );

                

                // all compacted rays are alive
                CUDA_CHECK_THROW(cudaMemsetAsync(workspace.ray_alive, true, n_rays_to_compact * sizeof(bool), stream));

                // swap the active and compact buffer indices
                std::swap(active_buf_idx, compact_buf_idx);

                // update n_rays_alive
                n_rays_alive = n_rays_to_compact;
            }
        }
        // increment the number of pixels filled
        n_pixels_filled += n_pixels_to_fill;
    }
};
