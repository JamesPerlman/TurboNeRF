#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include <tiny-cuda-nn/common.h>

#include "../utils/nerf-constants.cuh"
#include "../services/nerf-manager.cuh"
#include "../workspaces/training-workspace.cuh"

#include "../common.h"

/**
 * The Trainer class provides a simple interface for training a model.
 * An instance of Trainer is managed by the TrainingController.
 */

NRC_NAMESPACE_BEGIN

struct Trainer {
public:

	struct Context {
		const cudaStream_t& stream;
		TrainingWorkspace workspace;
		Dataset* dataset;
		NeRF* nerf;
		uint32_t batch_size;
		uint32_t n_rays_in_batch;
		uint32_t n_samples_in_batch;

		curandGenerator_t rng;

		Context(
			const cudaStream_t& stream,
			TrainingWorkspace& workspace,
			Dataset* dataset,
			NeRF* nerf,
			const uint32_t& batch_size
		)
			: stream(stream)
			, workspace(workspace)
			, dataset(dataset)
			, nerf(nerf)
			, batch_size(batch_size)
			, n_rays_in_batch(batch_size)
			, n_samples_in_batch(0)
		{
			// TODO: CURAND_ASSERT_SUCCESS
			curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
			curandGenerateSeeds(rng);
		};

		~Context() {
			curandDestroyGenerator(rng);
		};
	};

	// public methods
	void train_step(Context& ctx);
	void update_occupancy_grid(Context& ctx, const float& selection_threshold);
	void generate_next_training_batch(Context& ctx);
	void create_pixel_undistort_map(Context& ctx);
};

NRC_NAMESPACE_END
