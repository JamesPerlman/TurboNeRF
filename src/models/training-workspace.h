#pragma once

#include <stbi/stb_image.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "../common.h"
#include "bounding-box.cuh"
#include "camera.h"
#include "cascaded-occupancy-grid.cuh"
#include "ray.h"


NRC_NAMESPACE_BEGIN

// NeRFWorkspace?
// TODO: Make this a derived struct from RenderingWorkspace
struct TrainingWorkspace {
public:

	uint32_t batch_size;

	// arena properties
	BoundingBox* bounding_box;

	stbi_uc* image_data;
	tcnn::network_precision_t* density_input;
	tcnn::network_precision_t* density_output;

	tcnn::network_precision_t* color_output_r;
	tcnn::network_precision_t* color_output_g;
	tcnn::network_precision_t* color_output_b;

	float* random_floats;
	uint32_t* img_index;
	uint32_t* ray_index; // index of ray inside the batch
	uint32_t* pix_index; // index of randomly selected pixel in image

	uint32_t* n_steps;

	float* pix_r[2];
	float* pix_g[2];
	float* pix_b[2];
	float* pix_a[2];
	
	float* ray_r;
	float* ray_g;
	float* ray_b;
	float* ray_a;
	
	float* ori_x[2];
	float* ori_y[2];
	float* ori_z[2];

	float* dir_x[2];
	float* dir_y[2];
	float* dir_z[2];

	float* idir_x;
	float* idir_y;
	float* idir_z;

	float* ray_t; // t_mid
	float* ray_t0; // t_start
	float* ray_t1; // t_end

	uint32_t n_occupancy_grid_elements;
	CascadedOccupancyGrid* occupancy_grid;
	uint8_t* occupancy_grid_bitfield;

	// GPUMemory managed properties

	tcnn::GPUMemory<Camera> cameras;

	// constructor
	TrainingWorkspace() : arena_allocation() {};

	// member functions
	void enlarge(
		const cudaStream_t& stream,
		const uint32_t& n_images,
		const uint32_t& n_pixels_per_image,
		const uint32_t& n_channels_per_image,
		const uint32_t& n_samples_per_batch,
		const uint32_t& n_occupancy_grid_levels,
		const uint32_t& n_occupancy_grid_cells_per_dimension
	);

private:
	tcnn::GPUMemoryArena::Allocation arena_allocation;
};

NRC_NAMESPACE_END
