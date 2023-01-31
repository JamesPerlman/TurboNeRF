#include "../common.h"

namespace NeRFConstants {
    // min_step_size = sqrt(3.0f) / 1024.0f;
    constexpr float min_step_size = 0.00169145586f;

    constexpr float cone_angle = 1.0f / 256.0f;

    constexpr uint32_t batch_size = 2<<20;

    constexpr float min_transmittance = 1e-4f;
}
