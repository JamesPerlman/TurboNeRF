#include "../common.h"

#include "training-workspace.h"

using namespace nrc;
using namespace Eigen;
using network_precision_t = tcnn::network_precision_t;

void TrainingWorkspace::enlarge(
	const cudaStream_t& stream,
	const uint32_t& n_images,
	const uint32_t& n_pixels_per_image,
	const uint32_t& n_channels_per_image,
	const uint32_t& n_samples_per_batch,
	const uint32_t& n_occupancy_grid_levels,
	const uint32_t& n_occupancy_grid_cells_per_dimension
) {
	batch_size = tcnn::next_multiple(n_samples_per_batch, tcnn::batch_size_granularity);
	uint32_t n_cameras = tcnn::next_multiple(n_images, tcnn::batch_size_granularity);
	uint32_t n_pixel_elements = tcnn::next_multiple(n_channels_per_image * n_pixels_per_image * n_images, tcnn::batch_size_granularity);

	n_occupancy_grid_elements = CascadedOccupancyGrid::get_n_total_elements(n_occupancy_grid_levels, n_occupancy_grid_cells_per_dimension);
	uint32_t n_grid_bitfield_bytes = tcnn::next_multiple(n_occupancy_grid_elements / 8, tcnn::batch_size_granularity);

	auto data = tcnn::allocate_workspace_and_distribute<
		BoundingBox,				// bounding_box
		stbi_uc,					// image_data
		network_precision_t,		// density_input
		network_precision_t,		// color_output_rgb
		float,						// random_floats
		
		uint32_t,					// img_index
		uint32_t,					// pix_index

		uint32_t,					// n_steps
		
		float,						// pix_rgba
		float,						// ray_rgba

		float,						// ori_xyz
		float,						// dir_xyz
		float, 						// idir_xyz
		
		float,						// ray_t0
		float,						// ray_t1
		float, 						// pos_xyz

		CascadedOccupancyGrid,		// occupancy_grid
		uint8_t						// occupancy_grid_bitfield
	>(stream, &arena_allocation,
		1,							// bounding_box
		n_pixel_elements,			// img_data
		16 * batch_size,			// density_input
		3 * batch_size,				// color_output
		batch_size,					// random_floats

		batch_size,					// img_index
		2 * batch_size,				// pix_index (double buffer)

		2 * batch_size,				// n_steps (double buffer)
		
		2 * 4 * batch_size,			// pix_rgba (double buffer)
		4 * batch_size,				// ray_rgba
		
		2 * 3 * batch_size,			// ori_xyz (double buffer)
		2 * 3 * batch_size,			// dir_xyz (double buffer)
		2 * 3 * batch_size, 		// idir_xyz

		batch_size,					// ray_t0
		batch_size,					// ray_t1
		3 * batch_size, 			// pos_xyz

		1,							// occupancy_grid
		n_grid_bitfield_bytes		// occupancy_grid.bitfield

	);

	bounding_box = std::get<0>(data);
	image_data = std::get<1>(data);
	density_input = std::get<2>(data);
	
	color_output_rgb = std::get<3>(data);

	random_floats = std::get<4>(data);
	
	img_index = std::get<5>(data);
	pix_index = std::get<6>(data);

	n_steps[0] = std::get<7>(data);
	n_steps[1] = n_steps[0] + batch_size;
	
	// carefully note how double-buffered pointers are set up
	pix_rgba[0] = std::get<8>(data);
	pix_rgba[1] = pix_rgba[0] + 4 * batch_size;

	ray_rgba = std::get<9>(data);
	
	ori_xyz[0] = std::get<10>(data);
	ori_xyz[1] = ori_xyz[0] + 3 * batch_size;
	
	dir_xyz[0] = std::get<11>(data);
	dir_xyz[1] = dir_xyz[0] + 3 * batch_size;
	
	idir_xyz = std::get<12>(data);

	ray_t0 = std::get<13>(data);
	ray_t1 = std::get<14>(data);
	pos_xyz = std::get<15>(data);

	occupancy_grid = std::get<16>(data);
	occupancy_grid_bitfield = std::get<17>(data);
}
