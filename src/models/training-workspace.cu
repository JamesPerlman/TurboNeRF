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
		network_precision_t,		// color_output_r, color_output_g, color_outbut_b
		float,						// random_floats
		
		uint32_t,					// img_index
		uint32_t,					// pix_index

		uint32_t,					// n_steps
		
		float,						// pix_r, pix_g, pix_b, pix_a
		
		float,						// ori_x, ori_y, ori_z
		
		float,						// dir_x, dir_y, dir_z

		float, 						// idir_x, idir_y, idir_z
		
		float,						// ray_r, ray_g, ray_b, ray_a

		float,						// ray_t0
		float,						// ray_t1

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
		
		2 * 4 * batch_size,			// pix_r, pix_g, pix_b, pix_a (double buffer)
		
		2 * 3 * batch_size,			// ori_x, ori_y, ori_z (double buffer)
		
		2 * 3 * batch_size,			// dir_x, dir_y, dir_z (double buffer)

		2 * 3 * batch_size, 		// idir_x, idir_y, idir_z
		
		4 * batch_size,				// ray_r, ray_g, ray_b, ray_a

		batch_size,					// ray_t0
		batch_size,					// ray_t1

		1,							// occupancy_grid
		n_grid_bitfield_bytes		// occupancy_grid.bitfield

	);

	bounding_box = std::get<0>(data);
	image_data = std::get<1>(data);
	density_input = std::get<2>(data);
	
	color_output_r = std::get<3>(data);
	color_output_g = color_output_r + batch_size;
	color_output_b = color_output_g + batch_size;

	random_floats = std::get<4>(data);
	
	img_index = std::get<5>(data);
	pix_index = std::get<6>(data);

	n_steps = std::get<7>(data);
	
	// carefully note how double-buffered pointers are set up
	pix_r[0] = std::get<8>(data);
	pix_g[0] = pix_r[0] + batch_size;
	pix_b[0] = pix_g[0] + batch_size;
	pix_a[0] = pix_b[0] + batch_size;

	pix_r[1] = pix_a[0] + batch_size;
	pix_g[1] = pix_r[1] + batch_size;
	pix_b[1] = pix_g[1] + batch_size;
	pix_a[1] = pix_b[1] + batch_size;
	
	ori_x[0] = std::get<9>(data);
	ori_y[0] = ori_x[0] + batch_size;
	ori_z[0] = ori_y[0] + batch_size;

	ori_x[1] = ori_z[0] + batch_size;
	ori_y[1] = ori_x[1] + batch_size;
	ori_z[1] = ori_y[1] + batch_size;
	
	dir_x[0] = std::get<10>(data);
	dir_y[0] = dir_x[0] + batch_size;
	dir_z[0] = dir_y[0] + batch_size;

	dir_x[1] = dir_z[0] + batch_size;
	dir_y[1] = dir_x[1] + batch_size;
	dir_z[1] = dir_y[1] + batch_size;
	
	ray_r = std::get<11>(data);
	ray_g = ray_r + batch_size;
	ray_b = ray_g + batch_size;
	ray_a = ray_b + batch_size;

	idir_x = std::get<12>(data);
	idir_y = idir_x + batch_size;
	idir_z = idir_y + batch_size;

	ray_t0 = std::get<13>(data);
	ray_t1 = std::get<14>(data);

	occupancy_grid = std::get<15>(data);
	occupancy_grid_bitfield = std::get<16>(data);
}
