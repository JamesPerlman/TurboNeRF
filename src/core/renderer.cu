#include <tiny-cuda-nn/common.h>

#include "../models/camera.cuh"
#include "../utils/nerf-constants.cuh"
#include "../utils/parallel-utils.cuh"
#include "../utils/rendering-kernels.cuh"
#include "../utils/stream-compaction.cuh"

#include "renderer.cuh"

using namespace nrc;
using namespace tcnn;

void Renderer::enlarge_workspace_if_needed(
    Renderer::Context& ctx,
    const RenderRequest& request
) {
    const uint32_t new_render_area = request.output->width * request.output->height;

    // TODO: NeRF network property getters could be made into static members.
    // For now we assume all NeRFs have the same network configuration.
    NeRF& nerf = request.proxies[0]->nerfs[0];

    if (render_area != new_render_area) {
        ctx.workspace.enlarge(
            ctx.stream,
            request.output->width,
            request.output->height,
            ctx.batch_size,
            nerf.network.get_concat_buffer_width(),
            nerf.network.get_padded_output_width()
        );

        render_area = new_render_area;
    }
}

void Renderer::render(
    Renderer::Context& ctx,
    RenderRequest& request
) {
    RenderingWorkspace& workspace = ctx.workspace;
    
    // TODO: this should happen for all NeRFs
    NeRF& nerf = request.proxies[0]->nerfs[0];

    enlarge_workspace_if_needed(ctx, request);

    // workspace.camera = request.camera
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.camera,
            &request.camera,
            sizeof(Camera),
            cudaMemcpyHostToDevice,
            ctx.stream
        )
    );

    // workspace.bounding_box = nerf.bounding_box
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.bounding_box,
            &nerf.bounding_box,
            sizeof(BoundingBox),
            cudaMemcpyHostToDevice,
            ctx.stream
        )
    );

    // workspace.occupancy_grid = nerf.occupancy_grid
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.occupancy_grid,
            &nerf.occupancy_grid,
            sizeof(OccupancyGrid),
            cudaMemcpyHostToDevice,
            ctx.stream
        )
    );

    // calculate the number of pixels we need to fill
    uint32_t n_pixels = request.output->width * request.output->height;

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
            ctx.batch_size,
            n_pixels - n_pixels_filled
        );

        uint32_t n_rays = n_pixels_to_fill;

        // calculate the pixel indices to fill in this batch
        uint32_t pixel_start = n_pixels_filled;

        // generate rays for the pixels in this batch
        generate_rays_pinhole_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, ctx.stream>>>(
            n_rays,
            ctx.batch_size,
            workspace.bounding_box,
            workspace.camera,
            workspace.ray_origin[active_buf_idx],
            workspace.ray_dir[active_buf_idx],
            workspace.ray_idir[active_buf_idx],
            workspace.ray_t[active_buf_idx],
            workspace.ray_trans[active_buf_idx],
            workspace.ray_idx[active_buf_idx],
            workspace.ray_alive,
            workspace.ray_active[active_buf_idx],
            pixel_start
        );
        
        const float dt_min = NeRFConstants::min_step_size;
        const float dt_max = nerf.bounding_box.size_x * dt_min;
        const float cone_angle = NeRFConstants::cone_angle;

        march_rays_to_first_occupied_cell_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, ctx.stream>>>(
            n_rays,
            ctx.batch_size,
            workspace.occupancy_grid,
            workspace.bounding_box,
            dt_min,
            dt_max,
            cone_angle,

            // input buffers
            workspace.ray_dir[active_buf_idx],
            workspace.ray_idir[active_buf_idx],
            
            // dual-use buffers
            workspace.ray_alive,
            workspace.ray_origin[active_buf_idx],
            workspace.ray_t[active_buf_idx],

            // output buffers
            workspace.network_pos,
            workspace.network_dir,
            workspace.network_dt
        );

        // ray marching loop
        uint32_t n_rays_alive = n_rays;
        
        int n_steps = 0;

        // TODO: march rays to bounding box first
        while (n_rays_alive > 0) {

            // need to figure out how many rays can fit in this batch
            const uint32_t n_steps_per_ray = std::max(ctx.batch_size / n_rays_alive, (uint32_t)1);
            const uint32_t network_batch = tcnn::next_multiple(n_steps_per_ray * n_rays_alive, tcnn::batch_size_granularity);

            march_rays_and_generate_network_inputs_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, ctx.stream>>>(
                n_rays_alive,
                ctx.batch_size,
                n_steps_per_ray,
                network_batch,
                workspace.occupancy_grid,
                workspace.bounding_box,
                1.0f / nerf.bounding_box.size_x,
                dt_min,
                dt_max,
                cone_angle,

                // input buffers
                workspace.ray_origin[active_buf_idx],
                workspace.ray_dir[active_buf_idx],
                workspace.ray_idir[active_buf_idx],
                workspace.ray_alive,
                workspace.ray_active[active_buf_idx],
                workspace.ray_t[active_buf_idx],

                // output buffers
                workspace.ray_steps[active_buf_idx],
                workspace.network_pos,
                workspace.network_dir,
                workspace.network_dt
            );

            // query the NeRF network for the samples
            nerf.network.inference(
                ctx.stream,
                network_batch,
                workspace.network_pos,
                workspace.network_dir,
                workspace.network_concat,
                workspace.network_output
            );

            // save alpha in an buffer
            sigma_to_alpha_forward_kernel<<<n_blocks_linear(network_batch), n_threads_linear, 0, ctx.stream>>>(
                network_batch,
                workspace.network_output + 3 * network_batch,
                workspace.network_dt,
                workspace.sample_alpha
            );

            // accumulate these samples into the pixel colors
            composite_samples_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, ctx.stream>>>(
                n_rays_alive,
                network_batch,
                request.output->stride,

                // input buffers
                workspace.ray_active[active_buf_idx],
                workspace.ray_steps[active_buf_idx],
                workspace.ray_idx[active_buf_idx],
                workspace.network_output,
                workspace.sample_alpha,
                
                // output buffers
                workspace.ray_alive,
                workspace.ray_trans[active_buf_idx],
                request.output->rgba
            );

            n_steps += n_steps_per_ray;
            if (n_steps < NeRFConstants::n_steps_per_render_compaction) {
                continue;
            }

            // update how many rays are still alive
            const int n_rays_to_compact = count_true_elements(
                ctx.stream,
                n_rays_alive,
                workspace.ray_alive
            );

            // if no rays are alive, we can skip compositing
            if (n_rays_to_compact == 0) {
                break;
            }
            
            // check if compaction is required
            if (n_rays_to_compact < n_rays_alive / 2) {
                // get compacted ray indices
                generate_compaction_indices(
                    ctx.stream,
                    n_rays_alive,
                    workspace.ray_alive,
                    workspace.compact_idx
                );

                // compact ray properties via the indices
                compact_rays_kernel<<<n_blocks_linear(n_rays_to_compact), n_threads_linear, 0, ctx.stream>>>(
                    n_rays_to_compact,
                    ctx.batch_size,
                    workspace.compact_idx,

                    // input
                    workspace.ray_idx[active_buf_idx],
                    workspace.ray_active[active_buf_idx],
                    workspace.ray_t[active_buf_idx],
                    workspace.ray_origin[active_buf_idx],
                    workspace.ray_dir[active_buf_idx],
                    workspace.ray_idir[active_buf_idx],
                    workspace.ray_trans[active_buf_idx],

                    // output
                    workspace.ray_idx[compact_buf_idx],
                    workspace.ray_active[compact_buf_idx],
                    workspace.ray_t[compact_buf_idx],
                    workspace.ray_origin[compact_buf_idx],
                    workspace.ray_dir[compact_buf_idx],
                    workspace.ray_idir[compact_buf_idx],
                    workspace.ray_trans[compact_buf_idx]
                );

                // all compacted rays are alive
                CUDA_CHECK_THROW(cudaMemsetAsync(workspace.ray_alive, true, n_rays_to_compact * sizeof(bool), ctx.stream));

                // swap the active and compact buffer indices
                std::swap(active_buf_idx, compact_buf_idx);

                printf("compacted %d rays to %d rays\n", n_rays_alive, n_rays_to_compact);

                // update n_rays_alive
                n_rays_alive = n_rays_to_compact;

                n_steps = 0;
            }                                                                                                                                                                                                                                                                                                              
        }
        // increment the number of pixels filled
        n_pixels_filled += n_pixels_to_fill;
    }
};
