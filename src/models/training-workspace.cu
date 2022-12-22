#include "../common.h"

#include "training-workspace.h"

using namespace nrc;
using network_precision_t = tcnn::network_precision_t;

void TrainingWorkspace::enlarge(cudaStream_t stream, uint32_t n_pixels, uint32_t n_images, uint32_t training_batch_size) {

	batch_size = tcnn::next_multiple(training_batch_size, tcnn::batch_size_granularity);
	
	uint32_t n_rays = n_pixels * batch_size;
	uint32_t n_density_input_elements = 3 * n_pixels * batch_size;
	uint32_t n_density_output_elements = 16 * n_pixels * batch_size;
	uint32_t n_color_output_elements = 3 * n_pixels * batch_size;

	uint32_t n_batch_rng_elements = batch_size;
	uint32_t n_batch_rgb_elements = 3 * batch_size;
	uint32_t n_batch_ray_elements = batch_size;
	
	auto data = tcnn::allocate_workspace_and_distribute<
		network_precision_t,	// density inputs
		network_precision_t,	// density outputs
		network_precision_t,	// color outputs
		uint32_t,				// static image indices
		uint32_t,				// batch of random pixel indices
		float,					// batch of random floats
		float,					// batch of RGB inputs
		Ray						// batch of rays
	>(stream, &arena_allocation,
		n_density_input_elements,
		n_density_output_elements,
		n_color_output_elements,
		batch_size,
		n_batch_rng_elements,
		n_batch_rng_elements,
		n_batch_rgb_elements,
		n_batch_ray_elements
	);
	
	density_input	= std::get<0>(data);
	density_output	= std::get<1>(data);
	color_output	= std::get<2>(data);
	image_indices	= std::get<3>(data);
	pixel_indices	= std::get<4>(data);
	random_floats	= std::get<5>(data);
	rgb_batch		= std::get<6>(data);
	ray_batch		= std::get<7>(data);
}
