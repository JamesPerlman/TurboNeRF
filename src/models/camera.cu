#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "camera.h"


using namespace nrc;
using namespace Eigen;

// init_rays CUDA kernel
__global__ void init_rays_pinhole(
	uint32_t n_rays,
	Camera cam,
	Ray* rays_out
) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	int32_t idx = x * gridDim.x + y;

	if (idx >= n_rays) {
		return;
	}
	
	Ray ray = cam.get_ray_at_pixel_xy(x, y);
	
	rays_out[idx] = ray;
}

NRC_HOST_DEVICE Ray Camera::get_ray_at_pixel_xy(const uint32_t& x, const uint32_t& y) const {
	
	// uv ranges [-0.5f, 0.5f] in both dimensions, and is centered on this pixel
	Vector2f uv = Vector2f(
		float(x) / (float(pixel_dims.x()) + 0.5f),
		float(y) / (float(pixel_dims.y()) + 0.5f)
	) - Vector2f(0.5f, 0.5f);

	Vector2f uv_scaled = uv.cwiseProduct(sensor_size);
	Vector3f pix_pos = Vector3f(uv_scaled.x(), uv_scaled.y(), -1.0f);

	Vector3f ray_d = pix_pos;
	Vector3f ray_o = near * ray_d;

	return Ray{ ray_o, ray_d };
}
