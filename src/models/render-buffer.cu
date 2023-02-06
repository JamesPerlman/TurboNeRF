#include "render-buffer.cuh"
#include "../utils/gpu-image.cuh"

using namespace nrc;

void RenderBuffer::clear(const cudaStream_t& stream) {
    CUDA_CHECK_THROW(
        cudaMemsetAsync(rgba, 0, width * height * 4 * sizeof(float), stream)
    );
}

void RenderBuffer::save_image(const std::string& filename, const cudaStream_t& stream) {
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    save_buffer_to_image(stream, filename, rgba, width, height, 4, stride);
}

std::vector<float> RenderBuffer::get_image(const cudaStream_t& stream) {
    return save_buffer_to_memory(stream, rgba, width, height, 4, stride);
}
