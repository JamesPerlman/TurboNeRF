#include "nerf-rendering-controller.h"

using namespace nrc;

void NeRFRenderingController::request_render(
    const cudaStream_t& stream,
    const RenderRequest& request
) {
    printf("Render!!\n");
};
