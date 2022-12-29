#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common.h"

NRC_NAMESPACE_BEGIN

/* Instead of initializing an array of t-values to sample all at once, we sample after each raymarching iteration,
 * and accumulate the color of the ray at each step. This is more efficient than sampling all at once (I believe),
 * and also uses less memory.
 */
/*
__inline__ NRC_HOST_DEVICE void accumulate_colors(
	uint32_t n_rays,
	const float& max_steps,
    const float* __restrict__ network_r, const float* __restrict__ network_g, const float* __restrict__ network_b,
	const float* __restrict__ ray_o_x, const float* __restrict__ ray_o_y, const float* __restrict__ ray_o_z,
	const float* __restrict__ ray_d_x, const float* __restrict__ ray_d_y, const float* __restrict__ ray_d_z,
	float* __restrict__ ray_t,
	bool* __restrict__ ray_alive,
    float* __restrict__ r_out, float* __restrict__ g_out, float* __restrict__ b_out
) {
    
}
*/

NRC_NAMESPACE_END
