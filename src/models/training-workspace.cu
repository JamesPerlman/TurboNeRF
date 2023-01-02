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
		float,						// random_floats
		
		uint32_t,					// img_index
		uint32_t,					// pix_index

		uint32_t,					// ray_steps
		uint32_t,					// ray_steps_cumulative
		
		float,						// pix_rgba
		float,						// ray_rgba

		float,						// ray_origins
		float,						// sample_origins
		float,						// ray_dirs
		float,						// sample_dirs
		float, 						// ray_inv_dirs
		
		float,						// sample_t0
		float,						// sample_t1
		float,						// sample_dt
		float, 						// sample_positions

		CascadedOccupancyGrid,		// occupancy_grid
		uint8_t,					// occupancy_grid_bitfield

		float						// loss
	>(stream, &arena_allocation,
		1,							// bounding_box
		n_pixel_elements,			// img_data
		batch_size,					// random_floats

		batch_size,					// img_index
		batch_size,					// pix_index

		batch_size,					// ray_steps
		batch_size,					// ray_steps_cumulative
		
		4 * batch_size,				// pix_rgba
		4 * batch_size,				// ray_rgba
		
		3 * batch_size,				// ray_origins
		3 * batch_size,				// sample_origins
		3 * batch_size,				// ray_dirs
		3 * batch_size,				// sample_dirs
		3 * batch_size, 			// ray_inv_dirs

		batch_size,					// sample_t0
		batch_size,					// sample_t1
		batch_size,					// sample_dt
		3 * batch_size, 			// pos_xyz

		1,							// occupancy_grid
		n_grid_bitfield_bytes,		// occupancy_grid.bitfield

		batch_size					// loss

	);

	bounding_box = std::get<0>(data);
	image_data = std::get<1>(data);

	random_floats = std::get<2>(data);
	
	img_index = std::get<3>(data);
	pix_index = std::get<4>(data);

	ray_steps = std::get<5>(data);
	ray_steps_cumulative = std::get<6>(data);
	
	// carefully note how double-buffered pointers are set up
	pix_rgba = std::get<7>(data);

	ray_rgba = std::get<8>(data);
	
	ray_origins = std::get<9>(data);
	sample_origins = std::get<10>(data);
	
	ray_dirs = std::get<11>(data);
	sample_dirs = std::get<12>(data);
	
	ray_inv_dirs = std::get<13>(data);

	sample_t0 = std::get<14>(data);
	sample_t1 = std::get<15>(data);
	sample_dt = std::get<16>(data);

	sample_positions = std::get<17>(data);

	occupancy_grid = std::get<18>(data);
	occupancy_grid_bitfield = std::get<19>(data);

	loss = std::get<20>(data);
}
