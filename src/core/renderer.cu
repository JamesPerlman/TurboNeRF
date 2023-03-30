#include <tiny-cuda-nn/common.h>

#include "../models/camera.cuh"
#include "../models/ray-batch.cuh"
#include "../utils/nerf-constants.cuh"
#include "../utils/parallel-utils.cuh"
#include "../utils/rendering-kernels.cuh"
#include "../utils/stream-compaction.cuh"

#include "renderer.cuh"

using namespace turbo;
using namespace tcnn;

#define CHECK_IS_CANCELED(task) \
    if (task.is_canceled()) { \
        return; \
    }

void Renderer::prepare_for_rendering(
    Renderer::Context& ctx,
    const Camera& camera,
    const NeRF& nerf,
    const uint32_t& n_rays
) {
    cudaStream_t stream = ctx.stream;
    auto& workspace = ctx.workspace;

    if (workspace.n_rays != n_rays) {
        workspace.enlarge(
            stream,
            n_rays,
            ctx.batch_size,
            ctx.network.get_concat_buffer_width(),
            ctx.network.get_padded_output_width()
        );
    }

    // workspace.camera = request->camera
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.camera,
            &camera,
            sizeof(Camera),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    // workspace.bounding_box = nerf.bounding_box
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.bounding_box,
            &nerf.bounding_box,
            sizeof(BoundingBox),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    // workspace.occupancy_grid = nerf.occupancy_grid
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            workspace.occupancy_grid,
            &nerf.occupancy_grid,
            sizeof(OccupancyGrid),
            cudaMemcpyHostToDevice,
            stream
        )
    );
}

void Renderer::perform_task(
    Renderer::Context& ctx,
    RenderTask& task
) {
    RenderingWorkspace& workspace = ctx.workspace;
    
    // TODO: this should happen for all NeRFs
    NeRF* nerf = task.nerfs[0];
    
    cudaStream_t stream = ctx.stream;

    // double buffer indices
    int active_buf_idx = 0;
    int compact_buf_idx = 1;

    const int n_rays = task.n_rays;

    // ray.active = true
    CUDA_CHECK_THROW(
        cudaMemsetAsync(
            workspace.ray_active[active_buf_idx],
            true,
            n_rays * sizeof(bool),
            stream
        )
    );

    // ray.transmittance = 1.0
    float* __restrict__ T = workspace.ray_trans[active_buf_idx];
    parallel_for_gpu(stream, n_rays, [T] __device__ (uint32_t i) {
        T[i] = 1.0f;
    });

    // generate rays for the pixels in this batch
    RayBatch ray_batch{
        0,
        (int)n_rays,
        workspace.ray_origin[active_buf_idx],
        workspace.ray_dir[active_buf_idx],
        workspace.ray_idir[active_buf_idx],
        workspace.ray_t[active_buf_idx],
        workspace.ray_t_max[active_buf_idx],
        workspace.ray_trans[active_buf_idx],
        workspace.ray_idx[active_buf_idx],
        workspace.ray_active[active_buf_idx],
        workspace.ray_alive
    };
    
    // TODO: optimization here - add "clip to bbox" option to avoid extra computations for rays that don't intersect bbox
    task.batch_coordinator->generate_rays(
        workspace.camera,
        workspace.bounding_box,
        ray_batch,
        stream
    );

    const float dt_min = NeRFConstants::min_step_size;
    const float dt_max = nerf->bounding_box.size_x * dt_min;
    const float cone_angle = NeRFConstants::cone_angle;

    bool show_training_cameras = task.modifiers.properties.show_near_planes || task.modifiers.properties.show_far_planes;
    if (show_training_cameras) {
        // clear bg rgba first
        CUDA_CHECK_THROW(
            cudaMemsetAsync(
                workspace.bg_rgba,
                0,
                4 * n_rays * sizeof(float),
                stream
            )
        );

        // then draw clipping planes
        draw_training_img_clipping_planes_and_assign_t_max_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
            n_rays,
            n_rays,
            n_rays,
            nerf->dataset_ws.n_images,
            nerf->dataset_ws.image_dims,
            nerf->dataset_ws.n_pixels_per_image,
            task.modifiers.properties.show_near_planes,
            task.modifiers.properties.show_far_planes,
            nerf->dataset_ws.cameras,
            nerf->dataset_ws.image_data,
            workspace.ray_origin[active_buf_idx],
            workspace.ray_dir[active_buf_idx],
            workspace.ray_t_max[active_buf_idx],
            workspace.bg_rgba
        );
    }

    march_rays_to_first_occupied_cell_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        n_rays,
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
        workspace.ray_t_max[active_buf_idx],

        // output buffers
        workspace.network_pos,
        workspace.network_dir,
        workspace.network_dt
    );


    // ray marching loop
    uint32_t n_rays_alive = n_rays;
    int n_steps = 0;
    bool rgba_cleared = false;

    while (n_rays_alive > 0) {
        CHECK_IS_CANCELED(task);

        // need to figure out how many rays can fit in this batch
        const uint32_t n_steps_per_ray = std::max(ctx.batch_size / n_rays_alive, (uint32_t)1);
        const uint32_t network_batch = tcnn::next_multiple(n_steps_per_ray * n_rays_alive, tcnn::batch_size_granularity);

        march_rays_and_generate_network_inputs_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,
            n_rays,
            n_steps_per_ray,
            network_batch,
            workspace.occupancy_grid,
            workspace.bounding_box,
            1.0f / nerf->bounding_box.size_x,
            dt_min,
            dt_max,
            cone_angle,

            // input buffers
            workspace.ray_origin[active_buf_idx],
            workspace.ray_dir[active_buf_idx],
            workspace.ray_idir[active_buf_idx],
            workspace.ray_t_max[active_buf_idx],

            // dual-use buffers
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
        ctx.network.inference(
            stream,
            nerf->params,
            network_batch,
            nerf->aabb_scale(),
            workspace.network_pos,
            workspace.network_dir,
            workspace.network_concat,
            workspace.network_output
        );

        // save alpha in a buffer
        sigma_to_alpha_forward_kernel<<<n_blocks_linear(network_batch), n_threads_linear, 0, stream>>>(
            network_batch,
            workspace.network_output + 3 * network_batch,
            workspace.network_dt,
            workspace.sample_alpha
        );

        /**
         * It is best to clear RGBA right before we composite the first sample.
         * Just in case the task is canceled before we get a chance to draw anything.
         */

        if (!rgba_cleared) {
            // clear workspace.rgba
            CUDA_CHECK_THROW(cudaMemsetAsync(workspace.rgba, 0, 4 * n_rays * sizeof(float), stream));
            rgba_cleared = true;
        }

        // accumulate these samples into the pixel colors
        composite_samples_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,
            network_batch,
            n_rays,

            // input buffers
            workspace.ray_active[active_buf_idx],
            workspace.ray_steps[active_buf_idx],
            workspace.ray_idx[active_buf_idx],
            workspace.network_output,
            workspace.sample_alpha,
            
            // output buffers
            workspace.ray_alive,
            workspace.ray_trans[active_buf_idx],
            workspace.rgba
        );

        // We *could* render a partial result here
        // this progress is not very accurate, but it is fast.
        // float progress = (float)(n_rays - n_rays_alive) / (float)n_rays;
        // task.on_progress(progress);

        n_steps += n_steps_per_ray;
        if (n_steps < NeRFConstants::n_steps_per_render_compaction) {
            continue;
        }

        // update how many rays are still alive
        const int n_rays_to_keep = count_true_elements(
            stream,
            n_rays_alive,
            workspace.ray_alive
        );

        // if no rays are alive, we can skip compositing
        if (n_rays_to_keep == 0) {
            break;
        }
        
        // check if compaction is required
        if (n_rays_to_keep < n_rays_alive / 2) {
            CHECK_IS_CANCELED(task);
            
            // get compacted ray indices
            generate_compaction_indices(
                stream,
                n_rays_alive,
                workspace.ray_alive,
                workspace.compact_idx
            );

            // compact ray properties via the indices
            compact_rays_kernel<<<n_blocks_linear(n_rays_to_keep), n_threads_linear, 0, stream>>>(
                n_rays_to_keep,
                n_rays,
                workspace.compact_idx,

                // input
                workspace.ray_idx[active_buf_idx],
                workspace.ray_active[active_buf_idx],
                workspace.ray_t[active_buf_idx],
                workspace.ray_t_max[active_buf_idx],
                workspace.ray_origin[active_buf_idx],
                workspace.ray_dir[active_buf_idx],
                workspace.ray_idir[active_buf_idx],
                workspace.ray_trans[active_buf_idx],

                // output
                workspace.ray_idx[compact_buf_idx],
                workspace.ray_active[compact_buf_idx],
                workspace.ray_t[compact_buf_idx],
                workspace.ray_t_max[compact_buf_idx],
                workspace.ray_origin[compact_buf_idx],
                workspace.ray_dir[compact_buf_idx],
                workspace.ray_idir[compact_buf_idx],
                workspace.ray_trans[compact_buf_idx]
            );

            // all compacted rays are alive
            CUDA_CHECK_THROW(cudaMemsetAsync(workspace.ray_alive, true, n_rays_to_keep * sizeof(bool), stream));

            // swap the active and compact buffer indices
            std::swap(active_buf_idx, compact_buf_idx);

            // update n_rays_alive
            n_rays_alive = n_rays_to_keep;

            n_steps = 0;
        }                                                                                                                                                                                                                                                                                                              
    }

    if (show_training_cameras) {
        alpha_composite_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
            n_rays,
            n_rays,
            workspace.rgba,
            workspace.bg_rgba,
            workspace.rgba
        );
    }
};

void Renderer::write_to_target(
    Renderer::Context& ctx,
    RenderTask& task,
    RenderTarget* target
) {
    target->open_for_cuda_access(
        [&ctx, &task, target](float* rgba) {
            task.batch_coordinator->copy_packed(
                task.n_rays,
                {target->width, target->height},
                target->n_pixels(),
                ctx.workspace.rgba,
                rgba
            );
        },
        ctx.stream
    );
}
