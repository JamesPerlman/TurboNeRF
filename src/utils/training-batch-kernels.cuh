#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>

#include "../common.h"
#include "../core/occupancy-grid.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../utils/color-utils.cuh"
#include "../math/transform4f.cuh"

TURBO_NAMESPACE_BEGIN

/** This file contains helper kernels for generating rays and samples to fill the batch with data.
  */

__global__ void stbi_uchar_to_float(
    const uint32_t n_elements,
    const stbi_uc* __restrict__ src,
    float* __restrict__ dst
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        dst[idx] = (float)src[idx] / 255.0f;
    }
}

__global__ void resize_floats_to_uint32_with_max(
    const uint32_t n_elements,
    const float* __restrict__ floats,
    uint32_t* __restrict__ uints,
    const float range_max
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_elements) return;
    
    float resized_val = floats[idx] * range_max;
    uints[idx] = (uint32_t)resized_val;
}

/**
 * This device function generates a random pixel indices for each image.
 * Like image indices, we also want them to be in ascending order to preserve spatial locality.
 * 
 * To do this we slice the image into equal-sized chunks (not necessarily perfect rectangles, they might wrap).
 * For each chunk we generate a random pixel index.
 * 
 * Some of these parameters are floats because they will lead to a more accurate result.
 * Using integers could yield less coverage of the very last image.
 */

// actually it would probably be better if we put the random pixels in perfectly contiguous memory 

// THIS IS NOW CURRENTLY UNUSED
inline __device__ uint32_t random_pixel_index(
    const uint32_t global_chunk_index,
    const uint32_t n_pixels_per_image, // total number of pixels in each image
    const float n_chunks_per_image,
    const float chunk_size,
    const float random
) {
    // this is the index of the chunk we are generating a pixel index for, relative to the current image
    const float local_chunk_index = fmodf((float)global_chunk_index, n_chunks_per_image);

    // start of the current chunk, relative to the current image
    const float chunk_start = local_chunk_index * chunk_size;

    // end of the current chunk, relative to the current image
    const float chunk_end = chunk_start + chunk_size;

    // generate a random index between these two points
    return static_cast<uint32_t>(chunk_start + random * (chunk_end - chunk_start));
}

// generates rays and RGBs for training, assigns them to an array of contiguous data
__global__ void initialize_training_rays_and_pixels_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const uint32_t n_images,
    const uint32_t n_pixels_per_image,
    const int2 image_dimensions,
    const float n_rays_per_image,

    // input buffers
    const BoundingBox* __restrict__ bbox,
    const Camera* __restrict__ cameras,
    const stbi_uc* __restrict__ image_data,
    const float* __restrict__ random,

    // output buffers
    float* __restrict__ pix_rgba,
    float* __restrict__ ori_xyz,
    float* __restrict__ dir_xyz,
    float* __restrict__ ray_t,
    float* __restrict__ ray_t_max,
    bool* __restrict__ ray_alive
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) return;
    
    const size_t image_idx = (size_t)((float)i / n_rays_per_image);
    const float local_pixel_idx_f = (float)n_pixels_per_image * random[i];
    const float img_width_f = (float)image_dimensions.x;

    const float x = fmodf(local_pixel_idx_f, img_width_f);
    const float y = local_pixel_idx_f / img_width_f;

    const Camera cam = cameras[image_idx];
    
    Ray global_ray = cam.global_ray_at_pixel_xy(x, y);

    const float3 global_ori = global_ray.o;
    const float3 global_dir = global_ray.d;

    const float dir_x = global_dir.x;
    const float dir_y = global_dir.y;
    const float dir_z = global_dir.z;
    
    const float idir_x = 1.0f / dir_x;
    const float idir_y = 1.0f / dir_y;
    const float idir_z = 1.0f / dir_z;
    
    float tmin, tmax;
    const bool intersects_bbox = bbox->get_ray_t_intersections(
        global_ori.x, global_ori.y, global_ori.z,
        dir_x, dir_y, dir_z,
        idir_x, idir_y, idir_z,
        tmin, tmax
    );
    
    if (!intersects_bbox) {
        ray_alive[i] = false;
        return;
    }

    // calculate t_max
    const float t_max = fminf(cam.far - cam.near, tmax);
    const float t = fmaxf(0.0f, tmin + 1e-5f);

    if (t_max < t) {
        ray_alive[i] = false;
        return;
    }

    // local indices for contiguous buffers
    const uint32_t i_offset_0 = i;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;
    const uint32_t i_offset_3 = i_offset_2 + batch_size;
    
    // assign ground-truth pixel
    const size_t img_offset = (size_t)n_pixels_per_image * image_idx;
    const size_t local_pixel_idx = std::min((size_t)local_pixel_idx_f, (size_t)n_pixels_per_image - 1);
    const stbi_uc* __restrict__ pixel = image_data + 4 * (img_offset + local_pixel_idx);

    const float r = __srgb_to_linear((float)pixel[0] / 255.0f);
    const float g = __srgb_to_linear((float)pixel[1] / 255.0f);
    const float b = __srgb_to_linear((float)pixel[2] / 255.0f);
    const float a = (float)pixel[3] / 255.0f;

    pix_rgba[i_offset_0] = r;
    pix_rgba[i_offset_1] = g;
    pix_rgba[i_offset_2] = b;
    pix_rgba[i_offset_3] = a;

    // assign ray properties
    ori_xyz[i_offset_0] = global_ori.x;
    ori_xyz[i_offset_1] = global_ori.y;
    ori_xyz[i_offset_2] = global_ori.z;
    
    dir_xyz[i_offset_0] = dir_x;
    dir_xyz[i_offset_1] = dir_y;
    dir_xyz[i_offset_2] = dir_z;
    
    ray_t[i] = t;
    ray_t_max[i] = t_max;
    ray_alive[i] = true;
}

__global__ void deactivate_rays_with_alpha_threshold_kernel(
    const uint32_t n_rays,
    const uint32_t batch_size,
    const float alpha_threshold,
    const float selection_probability,
    const float* __restrict__ pix_rgba,
    const float* __restrict__ random,
    bool* __restrict__ ray_alive
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_rays) return;

    const float alpha = pix_rgba[3 * batch_size + i];
    const float random_value = random[i];

    if (alpha < alpha_threshold && random_value > selection_probability) {
        ray_alive[i] = false;
    }
}

// CONSIDER: move rays inside bounding box first?

__global__ void march_and_count_steps_per_ray_kernel(
    uint32_t n_rays,
    uint32_t batch_size,
    const BoundingBox* bbox,
    const OccupancyGrid* grid,
    const float cone_angle,
    const float dt_min,
    const float dt_max,
    
    // input buffers
    const float* __restrict__ dir_xyz,
    const float* __restrict__ ray_t_max,

    // output/mixed use buffers
    bool* __restrict__ ray_alive,
    float* __restrict__ ori_xyz,
    float* __restrict__ ray_t,
    uint32_t* __restrict__ n_steps // one per ray
) {
    // get thread index
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread is out of bounds
    if (i >= n_rays) return;

    if (!ray_alive[i]) {
        n_steps[i] = 0;
        return;
    };

    const uint32_t i_offset_0 = i;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;

    const float o_x = ori_xyz[i_offset_0];
    const float o_y = ori_xyz[i_offset_1];
    const float o_z = ori_xyz[i_offset_2];

    const float d_x = dir_xyz[i_offset_0];
    const float d_y = dir_xyz[i_offset_1];
    const float d_z = dir_xyz[i_offset_2];

    const float id_x = 1.0f / d_x;
    const float id_y = 1.0f / d_y;
    const float id_z = 1.0f / d_z;

    uint32_t n_steps_taken = 0;

    float t = ray_t[i];
    float t_max = ray_t_max[i];

    while (t < t_max) {
        const float x = o_x + t * d_x;
        const float y = o_y + t * d_y;
        const float z = o_z + t * d_z;

        if (!bbox->contains(x, y, z)) {
            break;
        }

        const float dt = grid->get_dt(t, cone_angle, dt_min, dt_max);
        const int grid_level = grid->get_grid_level_at(x, y, z, dt);

        if (grid->is_occupied_at(grid_level, x, y, z)) {

            t += dt;
            ++n_steps_taken;
        
        } else {

            t += grid->get_dt_to_next_voxel(
                x, y, z,
                d_x, d_y, d_z,
                id_x, id_y, id_z,
                dt_min,
                grid_level
            );
        
        }
    };

    if (n_steps_taken == 0) {
        ray_alive[i] = false;
    }

    n_steps[i] = n_steps_taken;
}

/**
 * This is just a helper function for setting normalized values for the ray before passing into the network
 * 
 */

inline __device__ void assign_normalized_ray_sample(
    const uint32_t& batch_size,
    const uint32_t& sample_offset,
    const uint32_t& n_steps_taken,
    const float& x, const float& y, const float& z,
    const float& dir_x, const float& dir_y, const float& dir_z,
    const float& dt,
    const float& inv_aabb_size,
    float* __restrict__ out_pos_xyz,
    float* __restrict__ out_dir_xyz,
    float* __restrict__ out_dt
) {
    
    const uint32_t step_offset_0 = sample_offset + n_steps_taken;
    const uint32_t step_offset_1 = step_offset_0 + batch_size;
    const uint32_t step_offset_2 = step_offset_1 + batch_size;

    // assign normalized network inputs
    out_dt[step_offset_0] = inv_aabb_size * dt;

    out_pos_xyz[step_offset_0] = tcnn::clamp(x * inv_aabb_size + 0.5f, 0.0f, 1.0f);
    out_pos_xyz[step_offset_1] = tcnn::clamp(y * inv_aabb_size + 0.5f, 0.0f, 1.0f);
    out_pos_xyz[step_offset_2] = tcnn::clamp(z * inv_aabb_size + 0.5f, 0.0f, 1.0f);

    out_dir_xyz[step_offset_0] = 0.5f * dir_x + 0.5f;
    out_dir_xyz[step_offset_1] = 0.5f * dir_y + 0.5f;
    out_dir_xyz[step_offset_2] = 0.5f * dir_z + 0.5f;
}

/**
 * This kernel has a few purposes:
 * 1. March rays through the occupancy grid and generate start/end intervals for each sample
 * 2. Compact other training buffers to maximize coalesced memory accesses
 */

__global__ void march_and_generate_network_positions_kernel(
    uint32_t n_rays,
    uint32_t batch_size,
    const BoundingBox* bbox,
    const float inv_aabb_size,
    const OccupancyGrid* grid,
    const float dt_min,
    const float dt_max,
    const float cone_angle,
    
    // input buffers
    const float* __restrict__ in_ori_xyz,
    const float* __restrict__ in_dir_xyz,
    const float* __restrict__ in_ray_t,
    const float* __restrict__ in_ray_t_max,
    const uint32_t* __restrict__ ray_offset,
    const bool* __restrict__ ray_alive,

    // dual-use buffers
    uint32_t* __restrict__ n_ray_steps,

    // output buffers
    float* __restrict__ out_pos_xyz,
    float* __restrict__ out_dir_xyz,
    float* __restrict__ out_dt,
    float* __restrict__ out_m_norm,
    float* __restrict__ out_dt_norm
) {
    // get thread index
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if thread is out of bounds
    if (i >= n_rays) return;

    if (!ray_alive[i]) return;
    
    // index properties for the loop
    const uint32_t n_steps = n_ray_steps[i];

    const uint32_t sample_offset = ray_offset[i];

    // References to input buffers

    const uint32_t i_offset_0 = i;
    const uint32_t i_offset_1 = i_offset_0 + batch_size;
    const uint32_t i_offset_2 = i_offset_1 + batch_size;

    const float o_x = in_ori_xyz[i_offset_0];
    const float o_y = in_ori_xyz[i_offset_1];
    const float o_z = in_ori_xyz[i_offset_2];

    const float d_x = in_dir_xyz[i_offset_0];
    const float d_y = in_dir_xyz[i_offset_1];
    const float d_z = in_dir_xyz[i_offset_2];
    
    const float id_x = 1.0f / d_x;
    const float id_y = 1.0f / d_y;
    const float id_z = 1.0f / d_z;
    
    // Perform raymarching

    float t = in_ray_t[i];

    const float t_min = t;
    float t_max = t;

    uint32_t n_steps_taken = 0;

    while (n_steps_taken < n_steps) {

        const float x = o_x + t * d_x;
        const float y = o_y + t * d_y;
        const float z = o_z + t * d_z;

        const float dt = grid->get_dt(t, cone_angle, dt_min, dt_max);

        if (!bbox->contains(x, y, z)) {
            out_m_norm[sample_offset + n_steps_taken] = t + 0.5f * dt;
            out_dt_norm[sample_offset + n_steps_taken] = dt;

            assign_normalized_ray_sample(
                batch_size, sample_offset, n_steps_taken,
                x, y, z,
                d_x, d_y, d_z,
                dt, inv_aabb_size,
                out_pos_xyz, out_dir_xyz, out_dt
            );

            ++n_steps_taken;
            
            continue;
        }

        const int grid_level = grid->get_grid_level_at(x, y, z, dt);

        if (grid->is_occupied_at(grid_level, x, y, z)) {

            out_m_norm[sample_offset + n_steps_taken] = t + 0.5f * dt;
            out_dt_norm[sample_offset + n_steps_taken] = dt;

            t += dt;
            t_max = t;

            assign_normalized_ray_sample(
                batch_size, sample_offset, n_steps_taken,
                x, y, z,
                d_x, d_y, d_z,
                dt, inv_aabb_size,
                out_pos_xyz, out_dir_xyz, out_dt
            );

            ++n_steps_taken;
        
        } else {

            // otherwise we need to find the next occupied cell
            t += grid->get_dt_to_next_voxel(
                x, y, z,
                d_x, d_y, d_z,
                id_x, id_y, id_z,
                dt_min,
                grid_level
            );
        }
    }

    // normalize t_mids
    const float t_span = std::max(t_max - t_min, 1e-6f);
    for (uint32_t i = 0; i < n_steps_taken; ++i) {
        out_m_norm[sample_offset + i] = tcnn::clamp((out_m_norm[sample_offset + i] - t_min) / t_span, 0.0f, 1.0f);
        out_dt_norm[sample_offset + i] /= t_span;
    }
}

TURBO_NAMESPACE_END
