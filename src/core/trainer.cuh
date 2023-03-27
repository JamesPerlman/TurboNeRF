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

TURBO_NAMESPACE_BEGIN

struct Trainer {
public:

	struct Context {
		const cudaStream_t& stream;
		TrainingWorkspace workspace;
		Dataset* dataset;
		NeRF* nerf;
		NerfNetwork network;
		uint32_t batch_size;
		uint32_t n_rays_in_batch;
		uint32_t n_samples_in_batch;

		curandGenerator_t rng;

		Context(
			const cudaStream_t& stream,
			TrainingWorkspace workspace,
			Dataset* dataset,
			NeRF* nerf,
			NerfNetwork network,
			uint32_t batch_size
		)
			: stream(stream)
			, workspace(std::move(workspace))
			, dataset(dataset)
			, nerf(nerf)
			, network(std::move(network))
			, batch_size(batch_size)
			, n_rays_in_batch(std::move(batch_size))
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
	float train_step(Context& ctx);
	uint32_t update_occupancy_grid(Context& ctx, const uint32_t& training_step);
	void generate_next_training_batch(Context& ctx);
};

TURBO_NAMESPACE_END
