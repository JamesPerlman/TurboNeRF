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
	
	uint32_t rng_size = batch_size;
	
	auto data = tcnn::allocate_workspace_and_distribute<
		Ray,
		network_precision_t,
		network_precision_t,
		network_precision_t,
		uint32_t,
		float
	>(stream, &arena_allocation,
		n_rays,
		n_density_input_elements,
		n_density_output_elements,
		n_color_output_elements,
		rng_size,
		rng_size
	);
	
	rays = std::get<0>(data);
	
	network_input = std::get<1>(data);
	network_output = std::get<2>(data);
	
	random_indices = std::get<4>(data);
	random_floats = std::get<5>(data);
}
