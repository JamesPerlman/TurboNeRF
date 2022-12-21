#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "camera.h"
#include "ray.h"


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
	// uv ranges [-0.5f, 0.5f] in both dimensions, and is centered on this pixel
	auto uv = Vector2f(
		float(x) / (float(cam.pixel_dims.x()) + 0.5f),
		float(y) / (float(cam.pixel_dims.y()) + 0.5f)
	) - Vector2f(0.5f, 0.5f);

	auto uv_scaled = uv.cwiseProduct(cam.sensor_size);
	auto pix_pos = Vector3f(uv_scaled.x(), uv_scaled.y(), 1.0f);
	
	auto cam_origin = cam.transform.block<3, 1>(0, 3);

	auto ray_d = pix_pos - cam_origin;
	auto ray_o = cam_origin + cam.near * ray_d;

	Ray ray = { ray_o, ray_d };
	rays_out[idx] = ray;
}
