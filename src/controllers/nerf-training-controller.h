# pragma once
#include <any>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <vector>

#include "../common.h"
#include "../core/nerf-network.cuh"
#include "../core/trainer.cuh"
#include "../models/bounding-box.cuh"
#include "../models/dataset.h"
#include "../models/nerf-proxy.cuh"
#include "../workspaces/training-workspace.cuh"

TURBO_NAMESPACE_BEGIN

struct NeRFTrainingController {

	struct TrainingMetrics {
		uint32_t step;
		uint32_t n_rays;
		uint32_t n_samples;
		float loss;

		std::map<std::string, std::any> as_map() const {
			return {
				{"step", step},
				{"n_rays", n_rays},
				{"n_samples", n_samples},
				{"loss", loss}
			};
		}
	};

	struct OccupancyGridMetrics {
		uint32_t n_occupied;
		uint32_t n_total;

		std::map<std::string, std::any> as_map() const {
			return {
				{"n_occupied", n_occupied},
				{"n_total", n_total}
			};
		}
	};

	// constructor
	NeRFTrainingController(
		NeRFProxy* proxy
	);

	NeRFTrainingController() = default;

	// public methods
	void setup_data(uint32_t batch_size = NeRFConstants::batch_size);

	void teardown();

	void reset_training();
	
	void load_images(std::function<void(int, int)> on_image_loaded = {});
	
	TrainingMetrics train_step();

	OccupancyGridMetrics update_occupancy_grid(const uint32_t& training_step);

	// training settings
	// TODO: put these in their own struct?
	float alpha_selection_threshold = 1.0f;
	float alpha_selection_probability = 1.0f;
	float min_step_size = NeRFConstants::min_step_size;
	
	NeRFProxy* proxy;

private:
	// private properties
	std::vector<Trainer::Context> contexts;
	Trainer trainer;

	bool _is_image_data_loaded = false;
	bool _is_training_memory_allocated = false;

	void update_dataset_if_necessary();
    std::vector<size_t> get_cuda_memory_allocated() const;
};

TURBO_NAMESPACE_END
