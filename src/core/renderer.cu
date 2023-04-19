#include <tiny-cuda-nn/common.h>

#include "../models/camera.cuh"
#include "../models/nerf-proxy.cuh"
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
    const std::vector<NeRF*>& nerfs,
    const uint32_t& n_rays
) {
    cudaStream_t stream = ctx.stream;
    auto& render_ws = ctx.render_ws;

    if (render_ws.n_rays != n_rays) {
        render_ws.enlarge(
            stream,
            n_rays,
            ctx.batch_size,
            ctx.network.get_concat_buffer_width(),
            ctx.network.get_padded_output_width()
        );
    }

    auto& scene_ws = ctx.scene_ws;
    const uint32_t n_nerfs = nerfs.size();

    if (scene_ws.n_nerfs != n_nerfs || scene_ws.n_rays != n_rays) {
        scene_ws.enlarge(
            stream,
            n_nerfs,
            n_rays
        );
    }

    // copy camera 
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            scene_ws.camera,
            &camera,
            sizeof(Camera),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    for (int i = 0; i < n_nerfs; i++) {
        const NeRF* nerf = nerfs[i];

        const NeRFProxy* proxy = nerf->proxy;

        // copy bounding boxes
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                scene_ws.bounding_boxes + i,
                &proxy->bounding_box,
                sizeof(BoundingBox),
                cudaMemcpyHostToDevice,
                stream
            )
        );

        // copy occupancy grids
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                scene_ws.occupancy_grids + i,
                &nerf->occupancy_grid,
                sizeof(OccupancyGrid),
                cudaMemcpyHostToDevice,
                stream
            )
        );

        // copy nerf transforms
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                scene_ws.nerf_transforms + i,
                &proxy->transform,
                sizeof(Transform4f),
                cudaMemcpyHostToDevice,
                stream
            )
        );
    }
}

void Renderer::perform_task(
    Renderer::Context& ctx,
    RenderTask& task
) {
    RenderingWorkspace& render_ws = ctx.render_ws;
    SceneWorkspace& scene_ws = ctx.scene_ws;
        
    cudaStream_t stream = ctx.stream;
    
    // double buffer indices
    int active_buf_idx = 0;
    int compact_buf_idx = 1;

    const int n_rays = task.n_rays;

    // ray.transmittance = 1.0
    float* __restrict__ T = render_ws.ray_trans[active_buf_idx];
    parallel_for_gpu(stream, n_rays, [T] __device__ (uint32_t i) {
        T[i] = 1.0f;
    });

    // generate rays for the pixels in this batch
    RayBatch ray_batch{
        0,
        (int)n_rays,
        render_ws.ray_origin[active_buf_idx],
        render_ws.ray_dir[active_buf_idx],
        render_ws.ray_tmax[active_buf_idx],
        render_ws.ray_trans[active_buf_idx],
        render_ws.ray_idx[active_buf_idx],
        render_ws.ray_alive
    };

    task.batch_coordinator->generate_rays(
        scene_ws.camera,
        ray_batch,
        stream
    );

    size_t n_nerfs = task.nerfs.size();

    const float dt_min = NeRFConstants::min_step_size;
    const float cone_angle = NeRFConstants::cone_angle;

    bool show_training_cameras = task.modifiers.properties.show_near_planes || task.modifiers.properties.show_far_planes;
    if (show_training_cameras) {
        NeRF* nerf = nullptr;
        for (auto& task_nerf : task.nerfs) {
            auto& task_proxy = task_nerf->proxy;
            const bool has_dataset = task_proxy->dataset.has_value();
            if (has_dataset) {
                nerf = task_nerf;
                // only one nerf can have a dataset for now
                break;
            }
        }

        if (nerf != nullptr) {
            // clear bg rgba first
            CUDA_CHECK_THROW(
                cudaMemsetAsync(
                    render_ws.bg_rgba,
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
                render_ws.ray_origin[active_buf_idx],
                render_ws.ray_dir[active_buf_idx],
                render_ws.ray_tmax[active_buf_idx],
                render_ws.bg_rgba
            );
        }
    }

    // this optimization only works if the camera rays travel in straight lines
    prepare_for_linear_raymarching_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
        n_rays,
        n_rays,
        n_nerfs,
        scene_ws.occupancy_grids,
        scene_ws.bounding_boxes,
        scene_ws.nerf_transforms,
        dt_min,
        cone_angle,

        // input buffers
        render_ws.ray_origin[active_buf_idx],
        render_ws.ray_dir[active_buf_idx],

        // dual-use buffers
        render_ws.ray_alive,
        render_ws.ray_tmax[active_buf_idx],

        // output buffers
        scene_ws.intersectors[active_buf_idx],
        scene_ws.ray_active[active_buf_idx],
        scene_ws.ray_t[active_buf_idx],
        scene_ws.ray_tmax[active_buf_idx]
    );

    // ray marching loop
    int n_steps = 0;
    bool rgba_cleared = false;
    
    uint32_t n_rays_processed = 0;
    
    uint32_t n_rays_alive = n_rays;

    while (n_rays_alive > 0) {
        CHECK_IS_CANCELED(task);

        const uint32_t n_steps_per_ray = std::max(ctx.batch_size / (3 * n_rays_alive * (uint32_t)n_nerfs), (uint32_t)1);
        const uint32_t network_batch = tcnn::next_multiple(n_steps_per_ray * n_rays_alive, tcnn::batch_size_granularity);

        march_rays_and_generate_network_inputs_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,
            n_nerfs,
            n_rays,
            network_batch,
            n_steps_per_ray,
            scene_ws.occupancy_grids,
            scene_ws.bounding_boxes,
            scene_ws.nerf_transforms,
            dt_min,
            cone_angle,

            // input buffers
            render_ws.ray_origin[active_buf_idx],
            render_ws.ray_dir[active_buf_idx],
            render_ws.ray_tmax[active_buf_idx],
            scene_ws.ray_tmax[active_buf_idx],
            scene_ws.intersectors[active_buf_idx],

            // dual-use buffers
            render_ws.ray_alive,
            scene_ws.ray_active[active_buf_idx],
            scene_ws.ray_t[active_buf_idx],

            // output buffers
            render_ws.n_steps_total,
            render_ws.sample_nerf_id,
            render_ws.network_pos[0],
            render_ws.network_dir[0],
            render_ws.network_dt
        );

        /**
         * Next we compact the network inputs for the active rays of each NeRF
         * Then we will query each respective NeRF network and composite the samples into the output buffer
         */

        for (int n = 0; n < n_nerfs; ++n) {
            // bool* rays_active_ptr = scene_ws.ray_active[active_buf_idx] + n * n_rays;
            uint32_t n_nerf_samples = n_nerfs == 1
                ? network_batch // minor optimization for single nerf
                : count_valued_elements(
                    stream,
                    n_rays_alive,
                    render_ws.sample_nerf_id,
                    n
                );

            if (n_nerf_samples == 0) {
                continue;
            }
            
            const uint32_t mini_network_batch = tcnn::next_multiple(n_nerf_samples, tcnn::batch_size_granularity);

            // no need to compact if the network batch is the same size as the number of samples

            float* network_pos;
            float* network_dir;
            int* net_compact_idx;
            bool compacted = false;

            if (mini_network_batch == network_batch) {
                
                network_pos = render_ws.network_pos[0];
                network_dir = render_ws.network_dir[0];
                net_compact_idx = nullptr;

            } else {
                
                generate_valued_compaction_indices(
                    stream,
                    n_rays_alive,
                    render_ws.sample_nerf_id,
                    n,
                    render_ws.net_compact_idx
                );

                // compact the network inputs for the active rays of this NeRF
                compact_network_inputs_kernel<<<n_blocks_linear(n_nerf_samples), n_threads_linear, 0, stream>>>(
                    n_nerf_samples,
                    network_batch,
                    mini_network_batch,
                    render_ws.net_compact_idx,

                    // input buffers
                    render_ws.network_pos[0],
                    render_ws.network_dir[0],

                    // output buffers
                    render_ws.network_pos[1],
                    render_ws.network_dir[1]
                );

                network_pos = render_ws.network_pos[1];
                network_dir = render_ws.network_dir[1];
                net_compact_idx = render_ws.net_compact_idx;

                compacted = true;
            }

            // query the NeRF network for the samples
            auto& nerf = task.nerfs[n];
            auto& proxy = nerf->proxy;

            // the data always flows to net_concat[1] and net_output[1], sorry for all the ternaries and conditionals
            ctx.network.inference(
                stream,
                nerf->params,
                mini_network_batch,
                (int)proxy->bounding_box.size(),
                network_pos,
                network_dir,
                render_ws.net_concat[compacted ? 0 : 1],
                render_ws.net_output[compacted ? 0 : 1]
            );

            // expand the network outputs back to the original network batch size
            if (compacted) {
                expand_network_outputs_kernel<<<n_blocks_linear(n_nerf_samples), n_threads_linear, 0, stream>>>(
                    n_nerf_samples,
                    mini_network_batch,
                    network_batch,
                    render_ws.net_compact_idx,

                    // input buffers
                    render_ws.net_output[0],
                    render_ws.net_concat[0],

                    // output buffers
                    render_ws.net_output[1],
                    render_ws.net_concat[1]
                );
            }
        }



        /**
         * It is best to clear RGBA right before we composite the first sample.
         * Just in case the task is canceled before we get a chance to draw anything.
         */

        if (!rgba_cleared) {
            // clear render_ws.rgba
            CUDA_CHECK_THROW(cudaMemsetAsync(render_ws.rgba, 0, 4 * n_rays * sizeof(float), stream));
            rgba_cleared = true;
        }

        // composite the samples into the output buffer

        // accumulate these samples into the pixel colors
        composite_samples_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,
            network_batch,
            n_rays,

            // input buffers
            n_steps_per_ray,
            render_ws.ray_idx[active_buf_idx],
            render_ws.network_dt,
            render_ws.net_output[1],
            render_ws.net_concat[1],
            render_ws.n_steps_total,

            // dual-use buffers
            render_ws.ray_trans[active_buf_idx],
            render_ws.rgba,

            // write-only buffers
            render_ws.ray_alive
        );

        // We *could* render a partial result here
        // this progress is not very accurate, but it is fast.
        // float progress = (float)(n_rays - n_rays_alive) / (float)n_rays;
        // task.on_progress(progress);

        n_steps += 1;
        if (n_steps < NeRFConstants::n_steps_per_render_compaction) {
            continue;
        }

        // update how many rays are still alive
        const int n_rays_to_keep = count_true_elements(
            stream,
            n_rays_alive,
            render_ws.ray_alive
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
                render_ws.ray_alive,
                render_ws.compact_idx
            );

            // compact ray properties via the indices
            compact_rays_kernel<<<n_blocks_linear(n_rays_to_keep), n_threads_linear, 0, stream>>>(
                n_rays_to_keep,
                n_nerfs,
                n_rays,
                render_ws.compact_idx,

                // input
                render_ws.ray_idx[active_buf_idx],
                scene_ws.ray_active[active_buf_idx],
                scene_ws.ray_t[active_buf_idx],
                scene_ws.ray_tmax[active_buf_idx],
                scene_ws.intersectors[active_buf_idx],
                render_ws.ray_tmax[active_buf_idx],
                render_ws.ray_origin[active_buf_idx],
                render_ws.ray_dir[active_buf_idx],
                render_ws.ray_trans[active_buf_idx],

                // output
                render_ws.ray_idx[compact_buf_idx],
                scene_ws.ray_active[compact_buf_idx],
                scene_ws.ray_t[compact_buf_idx],
                scene_ws.ray_tmax[compact_buf_idx],
                scene_ws.intersectors[compact_buf_idx],
                render_ws.ray_tmax[compact_buf_idx],
                render_ws.ray_origin[compact_buf_idx],
                render_ws.ray_dir[compact_buf_idx],
                render_ws.ray_trans[compact_buf_idx]
            );

            // all compacted rays are alive
            CUDA_CHECK_THROW(cudaMemsetAsync(render_ws.ray_alive, true, n_rays_to_keep * sizeof(bool), stream));

            // swap the active and compact buffer indices
            std::swap(active_buf_idx, compact_buf_idx);

            // update n_rays_alive
            n_rays_alive = n_rays_to_keep;

            n_steps = 0;
        }

        if (show_training_cameras) {
            alpha_composite_kernel<<<n_blocks_linear(n_rays), n_threads_linear, 0, stream>>>(
                n_rays,
                n_rays,
                render_ws.rgba,
                render_ws.bg_rgba,
                render_ws.rgba
            );
        }
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
                ctx.render_ws.rgba,
                rgba
            );
        },
        ctx.stream
    );
}
