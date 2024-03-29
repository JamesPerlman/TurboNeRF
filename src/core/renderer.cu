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

/**
 * This method prepares the scene for rendering.
 * It copies various data structures to the GPU.
 * 
 * TODO: Come up with a way to only copy the data that is needed. 
 */
void Renderer::prepare_for_rendering(
    Renderer::Context& ctx,
    const Camera& camera,
    const std::vector<NeRFRenderable>& renderables,
    const uint32_t& n_rays,
    bool always_copy_new_props
) {
    cudaStream_t stream = ctx.stream;

    enlarge_render_workspace_if_needed(ctx, renderables, n_rays);

    auto& scene_ws = ctx.scene_ws;
    const uint32_t n_nerfs = renderables.size();
    
    const bool needs_update = scene_ws.n_nerfs != n_nerfs || scene_ws.n_rays != n_rays;

    if (needs_update) {
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

    bool needs_copy = always_copy_new_props || needs_update;

    for (int i = 0; i < n_nerfs; i++) {
        const auto& renderable = renderables[i];
        const auto& proxy = renderable.proxy;
        const auto& nerf = proxy->nerfs[scene_ws.device_id];


        if (!proxy->can_render) {
            continue;
        }

        // copy updatable properties (these only get copied if their is_dirty flag is set)
        
        if (needs_copy || proxy->training_bbox.is_dirty()) {
            proxy->training_bbox.copy_to_device(scene_ws.training_bboxes + i, stream);
        }
        
        if (needs_copy || proxy->transform.is_dirty()) {
            proxy->transform.copy_to_device(scene_ws.nerf_transforms + i, stream);
        }

        // copy renderable bbox
        proxy->render_bbox.set_dirty(false);
        
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                scene_ws.render_bboxes + i,
                &renderable.bounding_box,
                sizeof(BoundingBox),
                cudaMemcpyHostToDevice,
                stream
            )
        );

        // copy occupancy grids
        // TODO: turn this into an UpdatableProperty
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                scene_ws.occupancy_grids + i,
                &nerf.occupancy_grid,
                sizeof(OccupancyGrid),
                cudaMemcpyHostToDevice,
                stream
            )
        );
    }
}

void Renderer::enlarge_render_workspace_if_needed(
    Renderer::Context& ctx,
    const std::vector<NeRFRenderable>& renderables,
    const uint32_t& n_rays
) {
    auto& render_ws = ctx.render_ws;

    size_t concat_buffer_width = 0;
    size_t padded_output_width = 0;

    // need to find the largest network sizes of all nerfs
    for (auto& renderable : renderables) {
        auto& proxy = renderable.proxy;
        auto& nerf = proxy->nerfs[render_ws.device_id];

        if (!proxy->can_render) {
            continue;
        }

        concat_buffer_width = std::max(concat_buffer_width, nerf.network.get_concat_buffer_width());
        padded_output_width = std::max(padded_output_width, nerf.network.get_padded_output_width());
    }

    if (render_ws.n_rays != n_rays ||
        render_ws.n_network_concat_elements != concat_buffer_width ||
        render_ws.n_network_output_elements != padded_output_width)
    {
        render_ws.enlarge(
            ctx.stream,
            n_rays,
            ctx.batch_size,
            concat_buffer_width,
            padded_output_width
        );

        // TODO: allow changes in appearance embeddings.  for now we just use ID = 0
        CUDA_CHECK_THROW(
            cudaMemsetAsync(
                render_ws.appearance_ids,
                0,
                render_ws.batch_size * sizeof(uint32_t),
                ctx.stream
            )
        );
    }
}

void Renderer::clear_rgba(
    Renderer::Context& ctx,
    RenderTask& task
) {
    CUDA_CHECK_THROW(
        cudaMemsetAsync(
            ctx.render_ws.rgba,
            0,
            4 * task.n_rays * sizeof(float),
            ctx.stream
        )
    );
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
    float* __restrict__ t = render_ws.ray_t[active_buf_idx];
    parallel_for_gpu(stream, n_rays, [T, t] __device__ (uint32_t i) {
        T[i] = 1.0f;
        t[i] = 0.0f;
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

    size_t n_nerfs = task.renderables.size();

    const float dt_min = ctx.min_step_size;
    const float cone_angle = NeRFConstants::cone_angle;

    bool show_training_cameras = task.modifiers.properties.show_near_planes || task.modifiers.properties.show_far_planes;
    if (show_training_cameras) {
        NeRF* nerf = nullptr;
        int nerf_idx = 0;
        for (auto& task_renderable : task.renderables) {
            auto& task_proxy = task_renderable.proxy;
            auto& task_nerf = task_proxy->nerfs[task.device_id];
            const bool has_dataset = task_proxy->dataset.has_value();
            if (has_dataset) {
                nerf = &task_proxy->nerfs[task.device_id];
                // only one nerf can have a dataset for now
                break;
            }
            ++nerf_idx;
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
                scene_ws.nerf_transforms + nerf_idx,
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
        scene_ws.render_bboxes,
        scene_ws.nerf_transforms,

        // dual-use buffers (read-write)
        render_ws.ray_alive,
        render_ws.ray_origin[active_buf_idx],
        render_ws.ray_dir[active_buf_idx],
        render_ws.ray_tmax[active_buf_idx]
    );

    // clear n_nerfs_for_sample
    CUDA_CHECK_THROW(
        cudaMemsetAsync(
            render_ws.n_nerfs_for_sample,
            0,
            ctx.batch_size * sizeof(int),
            stream
        )
    );

    // ray marching loop
    int n_steps = 0;
    bool rgba_cleared = false;
    
    uint32_t n_rays_alive = n_rays;

    while (n_rays_alive > 0) {
        CHECK_IS_CANCELED(task);

        const uint32_t n_steps_per_ray = std::max(ctx.batch_size / n_rays_alive, (uint32_t)1);
        const uint32_t n_samples_in_batch = n_steps_per_ray * n_rays_alive;
        const uint32_t network_batch = tcnn::next_multiple(n_samples_in_batch, tcnn::batch_size_granularity);

        const float dt_max = 8.0f * dt_min;
        march_rays_and_generate_global_sample_points_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,
            n_rays_alive,
            n_samples_in_batch,
            n_steps_per_ray,
            cone_angle,
            dt_min,
            dt_max,

            // input buffers (read-only)
            render_ws.ray_alive,
            render_ws.ray_origin[active_buf_idx],
            render_ws.ray_dir[active_buf_idx],

            // dual-use buffers (read-write)
            render_ws.ray_t[active_buf_idx],
            
            // output buffers (write-only)
            render_ws.sample_pos[0],
            render_ws.sample_dir[0],
            render_ws.sample_dt[0]
        );

        // clear out reusable sample buffers
        CUDA_CHECK_THROW(
            cudaMemsetAsync(
                render_ws.sample_rgba,
                0,
                4 * n_samples_in_batch * sizeof(float),
                stream
            )
        );

        /**
         * Next we localize the global sample points for each NeRF, and generate the normalized the network inputs.
         * Then we will query each respective NeRF network and blend the samples into a sample rgba buffer
         */
        
        for (int nerf_idx = 0; nerf_idx < n_nerfs; ++nerf_idx) {
            auto& renderable = task.renderables[nerf_idx];
            auto& proxy = renderable.proxy;
            NeRF* nerf = &proxy->nerfs[task.device_id];

            if (!proxy->can_render || !proxy->is_visible) {
                continue;
            }
            
            // generate the normalized network inputs for this NeRF
            filter_and_localize_samples_for_nerf_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
                n_rays_alive,
                n_samples_in_batch,
                n_steps_per_ray,
                proxy->transform.get().inverse(),
                renderable.bounding_box,

                // input buffers (read-only)
                render_ws.sample_pos[0],
                render_ws.sample_dir[0],
                render_ws.sample_dt[0],
                
                // output buffers (write-only)
                render_ws.n_nerfs_for_sample,
                render_ws.sample_valid,
                render_ws.sample_pos[1],
                render_ws.sample_dir[1],
                render_ws.sample_dt[1]
            );

            // apply effects

            BoundingBox render_bbox = proxy->render_bbox.get();
            
            for (const auto& spatial_effect : renderable.spatial_effects) {
                render_bbox = spatial_effect->get_bbox(render_bbox);

                spatial_effect->apply(
                    stream,
                    n_samples_in_batch,
                    n_samples_in_batch,
                    n_samples_in_batch,
                    render_ws.sample_valid,
                    render_ws.sample_pos[1],
                    render_ws.sample_pos[1]
                );
            }

            assign_normalized_network_inputs_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
                n_rays_alive,
                n_samples_in_batch,
                network_batch,
                n_steps_per_ray,
                1.0f / (proxy->training_bbox.get().size()),
                proxy->training_bbox.get(),
                nerf->occupancy_grid,

                // input buffers (read-only)
                render_ws.sample_pos[1],
                render_ws.sample_dir[1],
                render_ws.sample_dt[1],

                // dual-use buffers (read-write)
                render_ws.sample_valid,
                render_ws.n_nerfs_for_sample,
                render_ws.network_pos[0],
                render_ws.network_dir[0],
                render_ws.network_dt
            );

            uint32_t n_nerf_samples = count_valued_elements(
                    stream,
                    network_batch,
                    render_ws.sample_valid,
                    true
                );
            
            if (n_nerf_samples == 0) {
                continue;
            }

            const uint32_t mini_network_batch = tcnn::next_multiple(n_nerf_samples, tcnn::batch_size_granularity);

            float* network_pos;
            float* network_dir;
            bool compacted = false;

            // no need to compact if the network batch is the same size as the number of samples

            if (mini_network_batch == network_batch) {
                
                network_pos = render_ws.network_pos[0];
                network_dir = render_ws.network_dir[0];

            } else {

                generate_valued_compaction_indices(
                    stream,
                    network_batch,
                    render_ws.sample_valid,
                    true,
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

                compacted = true;
            }

            // query the NeRF network for the samples

            // the data always flows to net_concat[1] and net_output[1], sorry for all the ternaries and conditionals
            nerf->network.inference(
                stream,
                nerf->params,
                n_nerf_samples,
                mini_network_batch,
                (int)proxy->training_bbox.get().size(),
                render_ws.appearance_ids,
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

            // accumulate these samples into the sample_rgba buffer (they will be averaged in the composite step)
            accumulate_nerf_samples_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
                n_rays_alive,
                n_samples_in_batch,
                network_batch,
                n_steps_per_ray,

                // input buffers (read-only)
                render_ws.ray_alive,
                render_ws.sample_valid,
                render_ws.net_output[1],
                render_ws.net_concat[1],
                render_ws.network_dt,

                // dual-use buffers (read-write)
                render_ws.sample_rgba
            );
        }

        /**
         * It is best to clear RGBA right before we composite the first sample.
         * Just in case the task is canceled before we get a chance to draw anything.
         */

        if (!rgba_cleared) {
            // clear render_ws.rgba
            clear_rgba(ctx, task);
            rgba_cleared = true;
        }

        // composite the samples into the output buffer
        composite_samples_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,
            n_samples_in_batch,
            n_rays,
            n_steps_per_ray,

            // input buffers (read-only)
            render_ws.ray_idx[active_buf_idx],
            render_ws.n_nerfs_for_sample,
            render_ws.sample_rgba,

            // dual-use buffers (read-write)
            render_ws.ray_alive,
            render_ws.ray_trans[active_buf_idx],
            render_ws.rgba
        );

        // kill rays that have gone beyond the t max
        kill_terminated_rays_kernel<<<n_blocks_linear(n_rays_alive), n_threads_linear, 0, stream>>>(
            n_rays_alive,

            // input buffers (read-only)
            render_ws.ray_t[active_buf_idx],
            render_ws.ray_tmax[active_buf_idx],
            
            // dual-use buffers (read-write)
            render_ws.ray_alive
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
        const int n_rays_to_keep = count_valued_elements(
            stream,
            n_rays_alive,
            render_ws.ray_alive,
            true
        );

        // if no rays are alive, we end the loop
        if (n_rays_to_keep == 0) {
            break;
        }

        // check if compaction is required
        if (n_rays_to_keep < n_rays_alive / 2) {
            CHECK_IS_CANCELED(task);

            // get compacted ray indices
            generate_valued_compaction_indices(
                stream,
                n_rays_alive,
                render_ws.ray_alive,
                true,
                render_ws.compact_idx
            );

            // compact ray properties via the indices
            compact_rays_kernel<<<n_blocks_linear(n_rays_to_keep), n_threads_linear, 0, stream>>>(
                n_rays_to_keep,
                n_rays_alive,
                n_rays_to_keep,
                render_ws.compact_idx,

                // input
                render_ws.ray_idx[active_buf_idx],
                render_ws.ray_t[active_buf_idx],
                render_ws.ray_tmax[active_buf_idx],
                render_ws.ray_origin[active_buf_idx],
                render_ws.ray_dir[active_buf_idx],
                render_ws.ray_trans[active_buf_idx],

                // output
                render_ws.ray_idx[compact_buf_idx],
                render_ws.ray_t[compact_buf_idx],
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
