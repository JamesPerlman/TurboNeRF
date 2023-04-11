#pragma once

#include "../common.h"

namespace NeRFConstants {
    // min_step_size = sqrt(3.0f) / 1024.0f;
    constexpr float min_step_size = 0.00169145586f;

    constexpr float cone_angle = 1.0f / 256.0f;

    constexpr uint32_t batch_size = 2<<21;

    constexpr float min_transmittance = 1e-4f;

    constexpr float occupancy_decay = 0.95f;

	// This is adapted from the instant-NGP paper.  See page 15 on "Updating occupancy grids"
	// For some reason, the way the paper says it does not work for this implementation.
	// It seems to work with a threshold of 0.01, when the paper says to multiply by min_step_size.
    constexpr float occupancy_threshold = 0.01f;// * NeRFConstants::min_step_size;

    constexpr uint32_t n_steps_per_render_compaction = 64;

    constexpr float learning_rate_decay = 3.3e-5f;
}
