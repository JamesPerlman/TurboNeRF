
#include <device_launch_parameters.h>
#include <stbi/stb_image.h>

#include "rendering-kernels.cuh"
#include "../core/occupancy-grid.cuh"
#include "../math/geometric-intersections.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../utils/color-utils.cuh"
#include "../utils/common-network-kernels.cuh"
#include "../utils/nerf-constants.cuh"

using namespace tcnn;

TURBO_NAMESPACE_BEGIN


__global__ void draw_training_img_clipping_planes_and_assign_t_max_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const uint32_t out_rgba_stride,
    const uint32_t n_cameras,
    const int2 training_img_dims,
    const uint32_t n_pix_per_training_img,
    const bool show_near_planes,
    const bool show_far_planes,
    const Transform4f* __restrict__ nerf_transform,
    const Camera* __restrict__ cameras,
    const stbi_uc* __restrict__ train_img_data,
    const float* __restrict__ ray_ori,
    const float* __restrict__ ray_dir,
    float* __restrict__ ray_tmax,
    float* __restrict__ out_rgba_buf
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

    const Transform4f nerf_matrix = *nerf_transform;

    const uint32_t i_offset_0 = idx;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;

    const float3 ray_o = make_float3(
        ray_ori[i_offset_0],
        ray_ori[i_offset_1],
        ray_ori[i_offset_2]
    );

    const float3 ray_d = make_float3(
        ray_dir[i_offset_0],
        ray_dir[i_offset_1],
        ray_dir[i_offset_2]
    );

    float t_min = FLT_MAX;
    float2 t_min_uv;
    int t_min_cam_idx = -1;

    // we are looking for the minimum t-value of any plane that intersects this ray
    for (int i = 0; i < n_cameras; ++i) {
        const Camera cam = cameras[i];
        
        if (!cam.show_image_planes) {
            continue;
        }

        const Transform4f c2w = nerf_matrix * cam.transform;
        const Transform4f w2c = c2w.inverse();
        const float3 c2w_xyz = c2w.get_translation();
        const float3 r2{ c2w.m02, c2w.m12, c2w.m22 };
        const float m = l2_norm(r2);
        const float3 plane_normal = r2 / m;
        const float2 base_size{
            cam.resolution_f.x / cam.focal_length.x,
            cam.resolution_f.y / cam.focal_length.y
        };

        if (show_near_planes) {
            const float3 near_center = cam.near * plane_normal + c2w_xyz;
            const float2 near_size = (cam.near / m) * base_size;

            float t_near;
            float2 uv_near;

            bool intersects_near = ray_plane_intersection(
                ray_o,
                ray_d,
                near_center,
                plane_normal,
                near_size,
                w2c,
                uv_near,
                t_near
            );

            if (intersects_near && t_near < t_min) {
                t_min = t_near;
                t_min_uv = uv_near;
                t_min_cam_idx = i;
                continue;
            }
        }

        if (show_far_planes) {
            const float3 far_center = cam.far * plane_normal + c2w_xyz;
            const float2 far_size = (cam.far / m) * base_size;

            float t_far;
            float2 uv_far;

            bool intersects_far = ray_plane_intersection(
                ray_o,
                ray_d,
                far_center,
                plane_normal,
                far_size,
                w2c,
                uv_far,
                t_far
            );

            if (intersects_far && t_far < t_min) {
                t_min = t_far;
                t_min_uv = uv_far;
                t_min_cam_idx = i;
                continue;
            }
        }
    }

    // output pixel index
    const int out_idx_offset_0 = idx;
    const int out_idx_offset_1 = out_idx_offset_0 + (int)out_rgba_stride;
    const int out_idx_offset_2 = out_idx_offset_1 + (int)out_rgba_stride;
    const int out_idx_offset_3 = out_idx_offset_2 + (int)out_rgba_stride;
    
    // did we intersect anything?
    if (t_min_cam_idx > -1) {

        const int pix_ix = (int)(t_min_uv.x * (float)training_img_dims.x);
        const int pix_iy = (int)(t_min_uv.y * (float)training_img_dims.y);

        // clamp to the image bounds
        const size_t pix_x = (size_t)clamp(pix_ix, 0, training_img_dims.x - 1);
        const size_t pix_y = (size_t)clamp(pix_iy, 0, training_img_dims.y - 1);

        // get the pixel index
        const size_t train_pix_offset = n_pix_per_training_img * (size_t)t_min_cam_idx;
        const size_t pix_w = (size_t)training_img_dims.x;
        const size_t train_pix_idx = pix_y * pix_w + pix_x;

        const stbi_uc* train_rgba = train_img_data + 4 * (train_pix_offset + train_pix_idx);
        
        // write the pixel
        out_rgba_buf[out_idx_offset_0] = __srgb_to_linear((float)train_rgba[0] / 255.0f);
        out_rgba_buf[out_idx_offset_1] = __srgb_to_linear((float)train_rgba[1] / 255.0f);
        out_rgba_buf[out_idx_offset_2] = __srgb_to_linear((float)train_rgba[2] / 255.0f);
        out_rgba_buf[out_idx_offset_3] = (float)train_rgba[3] / 255.0f;

        // set t_max
        ray_tmax[idx] = t_min;
    } else {
        // clear output pixel
        out_rgba_buf[out_idx_offset_0] = 0.0f;
        out_rgba_buf[out_idx_offset_1] = 0.0f;
        out_rgba_buf[out_idx_offset_2] = 0.0f;
        out_rgba_buf[out_idx_offset_3] = 0.0f;
    }
}

__global__ void prepare_for_linear_raymarching_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const uint32_t n_nerfs,
    const OccupancyGrid* __restrict__ grids,
    const BoundingBox* __restrict__ render_bboxes,
    const Transform4f* __restrict__ transforms,
    
    // input buffers (read-only)
    const float* __restrict__ ray_ori,
    const float* __restrict__ ray_dir,

    // dual-use buffers (read/write)
    bool* __restrict__ ray_alive,
    float* __restrict__ ray_tmin,
    float* __restrict__ ray_tmax
) {
    // get thread index
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread is out of bounds or terminated
    if (i >= n_rays || !ray_alive[i]) return;

    // References to input buffers
    const uint32_t ray_offset_0 = i;
    const uint32_t ray_offset_1 = ray_offset_0 + batch_size;
    const uint32_t ray_offset_2 = ray_offset_1 + batch_size;

    const float3 global_ori{
        ray_ori[ray_offset_0],
        ray_ori[ray_offset_1],
        ray_ori[ray_offset_2]
    };

    const float3 global_dir{
        ray_dir[ray_offset_0],
        ray_dir[ray_offset_1],
        ray_dir[ray_offset_2]
    };

    // make sure this ray intersects some NeRF's bbox
    // tmin = float max
    float t_min = FLT_MAX;
    float t_max = 0.0f;

    int n_nerfs_hit = 0;

    for (int n = 0; n < n_nerfs; ++n) {

        const BoundingBox& bbox = render_bboxes[n];
        const Transform4f& transform = transforms[n];
        const Transform4f& itrans = transform.inverse();

        // we need to transform the ray into the NeRF's local space
        float3 local_ori = itrans * global_ori;
        float3 local_dir = itrans.mmul_ul3x3(global_dir);
        float3 local_idir = 1.0f / local_dir;

        float _tmin, _tmax;
        const bool intersects_bbox = (bbox.volume() > 0) && bbox.get_ray_t_intersections(local_ori, local_dir, local_idir, _tmin, _tmax);

        if (!intersects_bbox) {
            continue;
        }

        ++n_nerfs_hit;

        _tmin = fmaxf(_tmin, 1e-4f);

        if (_tmin < t_min) {
            t_min = _tmin;
        }

        if (_tmax > t_max) {
            t_max = _tmax;
        }
    }

    if (n_nerfs_hit == 0) {
        ray_alive[i] = false;
        return;
    }

    ray_tmin[i] = t_min;
    ray_tmax[i] = fminf(t_max, ray_tmax[i]);
}

// generate sample points in global space

__global__ void march_rays_and_generate_global_sample_points_kernel(
    const uint32_t n_rays,
    const uint32_t ray_batch_size,
    const uint32_t sample_stride,
    const uint32_t n_steps_per_ray,
    const float dt,

    // input buffers (read-only)
    const bool* __restrict__ ray_alive,
    const float* __restrict__ ray_tmax,
    const float* __restrict__ ray_ori,
    const float* __restrict__ ray_dir,

    // dual-use buffers (read-write)
    float* __restrict__ ray_t,

    // output buffers (write-only)
    float* __restrict__ sample_t,
    float* __restrict__ sample_pos,
    float* __restrict__ sample_dir,
    float* __restrict__ sample_dt,
    int* __restrict__ n_nerfs_for_sample
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread is out of bound
    if (i >= n_rays || !ray_alive[i]) return;

    // References to input buffers
    const uint32_t ray_offset_0 = i;
    const uint32_t ray_offset_1 = ray_offset_0 + ray_batch_size;
    const uint32_t ray_offset_2 = ray_offset_1 + ray_batch_size;

    const float o_x = ray_ori[ray_offset_0];
    const float o_y = ray_ori[ray_offset_1];
    const float o_z = ray_ori[ray_offset_2];

    const float d_x = ray_dir[ray_offset_0];
    const float d_y = ray_dir[ray_offset_1];
    const float d_z = ray_dir[ray_offset_2];

    
    // Perform raymarching
    float tmax = ray_tmax[ray_offset_0];
    float t = ray_t[ray_offset_0];

    for (uint32_t s = 0; s < n_steps_per_ray; ++s) {
        const float x = o_x + t * d_x;
        const float y = o_y + t * d_y;
        const float z = o_z + t * d_z;

        const uint32_t sample_offset_0 = n_rays * s + i;
        const uint32_t sample_offset_1 = sample_offset_0 + sample_stride;
        const uint32_t sample_offset_2 = sample_offset_1 + sample_stride;

        sample_t[sample_offset_0] = t;

        sample_pos[sample_offset_0] = x;
        sample_pos[sample_offset_1] = y;
        sample_pos[sample_offset_2] = z;

        sample_dir[sample_offset_0] = d_x;
        sample_dir[sample_offset_1] = d_y;
        sample_dir[sample_offset_2] = d_z;

        sample_dt[sample_offset_0] = dt;

        n_nerfs_for_sample[sample_offset_0] = 0;

        t += dt;
    }

    ray_t[ray_offset_0] = t;
}

// For global sample points, determine which ones hit a particular NeRF
__global__ void filter_and_assign_network_inputs_for_nerf_kernel(
    const uint32_t n_rays,
    const uint32_t sample_stride,
    const uint32_t network_stride,
    const uint32_t n_steps_per_ray,
    const Transform4f world_to_nerf,
    const float inv_nerf_scale,
    const BoundingBox render_bbox,
    const BoundingBox training_bbox,
    const OccupancyGrid occupancy_grid,

    // input buffers (read-only)
    const float* __restrict__ sample_pos,
    const float* __restrict__ sample_dir,
    const float* __restrict__ sample_dt,

    // output buffers (write-only)
    int* __restrict__ n_nerfs_per_sample,
    bool* __restrict__ sample_valid,
    float* __restrict__ network_pos,
    float* __restrict__ network_dir
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) return;

    for (uint32_t s = 0; s < n_steps_per_ray; ++s) {

        const uint32_t sample_offset_0 = n_rays * s + i;
        const uint32_t sample_offset_1 = sample_offset_0 + sample_stride;
        const uint32_t sample_offset_2 = sample_offset_1 + sample_stride;

        const float3 global_pos{
            sample_pos[sample_offset_0],
            sample_pos[sample_offset_1],
            sample_pos[sample_offset_2]
        };

        const float3 local_pos = world_to_nerf * global_pos;

        const bool is_valid = render_bbox.contains(local_pos);

        if (!is_valid) {
            sample_valid[sample_offset_0] = false;
            continue;
        }

        // apply effects here

        const float3 global_dir{
            sample_dir[sample_offset_0],
            sample_dir[sample_offset_1],
            sample_dir[sample_offset_2]
        };
        
        const float3 local_dir = world_to_nerf.mmul_ul3x3(global_dir);

        const float dt = inv_nerf_scale * sample_dt[sample_offset_0];
        
        const int grid_level = occupancy_grid.get_grid_level_at(local_pos, dt);

        if (!occupancy_grid.is_occupied_at(local_pos, grid_level)) {
            sample_valid[sample_offset_0] = false;
            continue;
        }

        const float3 normalized_pos = training_bbox.pos_to_unit(local_pos);

        n_nerfs_per_sample[sample_offset_0] += 1;

        const uint32_t network_offset_0 = sample_offset_0;
        const uint32_t network_offset_1 = network_offset_0 + network_stride;
        const uint32_t network_offset_2 = network_offset_1 + network_stride;

        network_pos[network_offset_0] = tcnn::clamp(normalized_pos.x, 0.0f, 1.0f);
        network_pos[network_offset_1] = tcnn::clamp(normalized_pos.y, 0.0f, 1.0f);
        network_pos[network_offset_2] = tcnn::clamp(normalized_pos.z, 0.0f, 1.0f);

        network_dir[network_offset_0] = 0.5f * local_dir.x + 0.5f;
        network_dir[network_offset_1] = 0.5f * local_dir.y + 0.5f;
        network_dir[network_offset_2] = 0.5f * local_dir.z + 0.5f;

        sample_valid[sample_offset_0] = true;
    }
}

__global__ void accumulate_nerf_samples_kernel(
    const uint32_t n_rays,
    const uint32_t sample_stride,
    const uint32_t network_stride,
    const uint32_t n_steps_per_ray,

    // input buffers (read-only)
    bool* __restrict__ ray_alive,
    bool* __restrict__ sample_valid,
    const float* __restrict__ sample_dt,
    const tcnn::network_precision_t* __restrict__ network_rgb,
    const tcnn::network_precision_t* __restrict__ network_density,

    // dual-use buffers (read-write)
    float* __restrict__ sample_rgba
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays || !ray_alive[i]) return;

    for (uint32_t s = 0; s < n_steps_per_ray; ++s) {

        const uint32_t sample_offset_0 = n_rays * s + i;
        const uint32_t sample_offset_1 = sample_offset_0 + sample_stride;
        const uint32_t sample_offset_2 = sample_offset_1 + sample_stride;
        const uint32_t sample_offset_3 = sample_offset_2 + sample_stride;

        if (!sample_valid[sample_offset_0]) continue;

        const uint32_t network_offset_0 = sample_offset_0;
        const uint32_t network_offset_1 = network_offset_0 + network_stride;
        const uint32_t network_offset_2 = network_offset_1 + network_stride;

        const float network_r = (float)network_rgb[network_offset_0];
        const float network_g = (float)network_rgb[network_offset_1];
        const float network_b = (float)network_rgb[network_offset_2];

        const float network_a = density_to_alpha(
            network_density[network_offset_0],
            sample_dt[sample_offset_0]
        );

        // accumulate weighted samples
        sample_rgba[sample_offset_0] += network_r * network_a;
        sample_rgba[sample_offset_1] += network_g * network_a;
        sample_rgba[sample_offset_2] += network_b * network_a;
        sample_rgba[sample_offset_3] += network_a;
    }
}

__global__ void composite_samples_kernel(
    const uint32_t n_rays,
    const uint32_t sample_stride,
    const uint32_t output_stride,
    const uint32_t n_steps_per_ray,

    // read-only
    const int* __restrict__ ray_idx,
    const int* __restrict__ n_nerfs_for_sample,
    const float* __restrict__ sample_rgba,

    // read/write
    bool* __restrict__ ray_alive,
    float* __restrict__ ray_trans,
    float* __restrict__ output_rgba
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays || !ray_alive[i]) return;

    // accumulated pixel colors
    float out_r = 0.0f;
    float out_g = 0.0f;
    float out_b = 0.0f;
    float out_a = 0.0f;

    float trans = ray_trans[i];
    
    for (uint32_t s = 0; s < n_steps_per_ray; ++s) {
            
        const uint32_t sample_offset_0 = n_rays * s + i;
        const uint32_t sample_offset_1 = sample_offset_0 + sample_stride;
        const uint32_t sample_offset_2 = sample_offset_1 + sample_stride;
        const uint32_t sample_offset_3 = sample_offset_2 + sample_stride;

        const uint32_t n_nerfs = n_nerfs_for_sample[sample_offset_0];

        if (n_nerfs == 0) {
            continue;
        }

        // normalize accumulated color

        const float sample_r = sample_rgba[sample_offset_0];
        const float sample_g = sample_rgba[sample_offset_1];
        const float sample_b = sample_rgba[sample_offset_2];
        const float sample_a = sample_rgba[sample_offset_3];

        const float inv_a = (sample_a > 0.0001f) ? (1.0f / sample_a) : 0.0f;

        const float r = sample_r * inv_a;
        const float g = sample_g * inv_a;
        const float b = sample_b * inv_a;
        const float a = fminf(sample_a, 1.0f); // alpha is additive

        // accumulate

        const float weight = a * trans;

        out_r += weight * r;
        out_g += weight * g;
        out_b += weight * b;
        out_a += weight;

        // update transmittance

        trans *= 1.0f - a;

        // threshold
        
        if (trans < NeRFConstants::min_transmittance) {
            ray_alive[i] = false;
            break;
        }
    }

    ray_trans[i] = trans;
    
    // pixel indices
    const uint32_t idx_offset_0 = (uint32_t)ray_idx[i];
    const uint32_t idx_offset_1 = idx_offset_0 + output_stride;
    const uint32_t idx_offset_2 = idx_offset_1 + output_stride;
    const uint32_t idx_offset_3 = idx_offset_2 + output_stride;
    
    output_rgba[idx_offset_0] += out_r;
    output_rgba[idx_offset_1] += out_g;
    output_rgba[idx_offset_2] += out_b;
    output_rgba[idx_offset_3] += out_a;
}

__global__ void kill_terminated_rays_kernel(
    const uint32_t n_rays,

    // input buffers (read-only)
    const float* __restrict__ ray_t,
    const float* __restrict__ ray_tmax,
    
    // dual-use buffers (read-write)
    bool* __restrict__ ray_alive
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays || !ray_alive[i]) return;

    if (ray_t[i] > ray_tmax[i]) {
        ray_alive[i] = false;
    }
}

// sample compaction
__global__ void compact_network_inputs_kernel(
    const uint32_t n_compacted_samples,
    const uint32_t old_batch_size,
    const uint32_t new_batch_size,
    const int* __restrict__ indices,

    // input buffers (read-only)
    const float* __restrict__ in_network_pos,
    const float* __restrict__ in_network_dir,

    // output buffers (write-only)
    float* __restrict__ out_network_pos,
    float* __restrict__ out_network_dir
) {
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= n_compacted_samples) return;

    // 3-component buffers
    int c_offset = c_idx;
    int e_offset = indices[c_idx];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        out_network_pos[c_offset] = in_network_pos[e_offset];
        out_network_dir[c_offset] = in_network_dir[e_offset];

        c_offset += new_batch_size;
        e_offset += old_batch_size;
    }
}

// sample re-expansion
__global__ void expand_network_outputs_kernel(
    const uint32_t n_compacted_samples,
    const uint32_t old_batch_size,
    const uint32_t new_batch_size,
    const int* __restrict__ indices,

    // input buffers (read-only)
    const tcnn::network_precision_t* __restrict__ in_network_rgb,
    const tcnn::network_precision_t* __restrict__ in_network_density,

    // output buffers (write-only)
    tcnn::network_precision_t* __restrict__ out_network_rgb,
    tcnn::network_precision_t* __restrict__ out_network_density
) {
    const int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= n_compacted_samples) return;

    int c_offset = c_idx;
    int e_offset = indices[c_idx];

    out_network_density[e_offset] = in_network_density[c_offset];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        out_network_rgb[e_offset] = in_network_rgb[c_offset];

        c_offset += old_batch_size;
        e_offset += new_batch_size;
    }
}

// ray compaction
__global__ void compact_rays_kernel(
    const int n_compacted_rays,
    const int old_batch_size,
    const int new_batch_size,
    const int* __restrict__ indices,

    // input buffers (read-only)
    const int* __restrict__ in_idx, // this is the ray-pixel index
    const float* __restrict__ in_ray_t,
    const float* __restrict__ in_ray_tmax,
    const float* __restrict__ in_ori,
    const float* __restrict__ in_dir,
    const float* __restrict__ in_trans,

    // compacted output buffers (write-only)
    int* __restrict__ out_idx,
    float* __restrict__ out_ray_t,
    float* __restrict__ out_ray_tmax,
    float* __restrict__ out_ori,
    float* __restrict__ out_dir,
    float* __restrict__ out_trans
) {
    // compacted index is the index to write to
    int c_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_idx >= n_compacted_rays) return;

    // expanded index is the index to read from
    int e_idx = indices[c_idx];

    // 1-component buffers (global)
    out_idx[c_idx]              = in_idx[e_idx];
    out_ray_t[c_idx]            = in_ray_t[e_idx];
    out_ray_tmax[c_idx]         = in_ray_tmax[e_idx];
    out_trans[c_idx]            = in_trans[e_idx];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        out_ori[c_idx] = in_ori[e_idx];
        out_dir[c_idx] = in_dir[e_idx];

        c_idx += new_batch_size;
        e_idx += old_batch_size;
    }
}

// Thank you Copilot + GPT-4!
__global__ void alpha_composite_kernel(
    const uint32_t n_pixels,
    const uint32_t img_stride,
    const float* rgba_fg,
    const float* rgba_bg,
    float* rgba_out
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_pixels) return;

    const uint32_t idx_offset_0 = idx;
    const uint32_t idx_offset_1 = idx_offset_0 + img_stride;
    const uint32_t idx_offset_2 = idx_offset_1 + img_stride;
    const uint32_t idx_offset_3 = idx_offset_2 + img_stride;

    const float fg_r = rgba_fg[idx_offset_0];
    const float fg_g = rgba_fg[idx_offset_1];
    const float fg_b = rgba_fg[idx_offset_2];
    const float fg_a = rgba_fg[idx_offset_3];

    const float bg_r = rgba_bg[idx_offset_0];
    const float bg_g = rgba_bg[idx_offset_1];
    const float bg_b = rgba_bg[idx_offset_2];
    const float bg_a = rgba_bg[idx_offset_3];

    const float out_a = fg_a + bg_a * (1.0f - fg_a);
    rgba_out[idx_offset_3] = out_a;

    if (out_a > 0.0f) {
        const float out_r = (fg_r * fg_a + bg_r * bg_a * (1.0f - fg_a)) / out_a;
        const float out_g = (fg_g * fg_a + bg_g * bg_a * (1.0f - fg_a)) / out_a;
        const float out_b = (fg_b * fg_a + bg_b * bg_a * (1.0f - fg_a)) / out_a;

        rgba_out[idx_offset_0] = out_r;
        rgba_out[idx_offset_1] = out_g;
        rgba_out[idx_offset_2] = out_b;
    }
}

TURBO_NAMESPACE_END
