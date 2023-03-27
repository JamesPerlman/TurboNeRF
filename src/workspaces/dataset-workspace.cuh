#pragma once

#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"
#include "../core/occupancy-grid.cuh"
#include "../models/bounding-box.cuh"
#include "../models/camera.cuh"
#include "../models/ray.h"
#include "workspace.cuh"


TURBO_NAMESPACE_BEGIN

// NeRFWorkspace?
// TODO: Make this a derived struct from RenderingWorkspace
struct DatasetWorkspace: Workspace {

    using Workspace::Workspace;

    Camera* cameras;
	BoundingBox* bounding_box;
	stbi_uc* image_data;

	// primitives
	size_t n_images;
	size_t n_pixels_per_image;
    int2 image_dims;

	// member functions
	void enlarge(
		const cudaStream_t& stream,
		const size_t& n_images,
        const int2& image_dims
	) {
		free_allocations();

		this->n_images = n_images;
        this->image_dims = image_dims;
		this->n_pixels_per_image = image_dims.x * image_dims.y;
		
		size_t n_pixel_elements = 4 * n_pixels_per_image * n_images;

		bounding_box 	= allocate<BoundingBox>(stream, 1);
        cameras 		= allocate<Camera>(stream, n_images);
		image_data 		= allocate<stbi_uc>(stream, n_pixel_elements);
}
};

TURBO_NAMESPACE_END
