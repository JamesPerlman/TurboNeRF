#pragma once

#include <memory>
#include <tiny-cuda-nn/common.h>
#include "../controllers/nerf-training-controller.h"
#include "../utils/nerf-constants.cuh"
#include "../common.h"
#include "nerf-manager.cuh"

/**
 * The Trainer class provides a simple interface for training a model.
 * 
 * It is responsible for managing a NeRFTrainingController.
 * 
 * Some discussion/TODOs:
 * 
 * This class should probably serve as the link between the controller and its data.
 * TODO: Remove integrated workspace from the controller, pass it in as an arg from this class.
 * Trainer is responsible for managing CUDA streams.
 * It will also be responsible for managing the training process on multiple GPUs.
 * 
 */

NRC_NAMESPACE_BEGIN

struct Trainer {
// configurable properties
    READONLY_PROPERTY(uint32_t, current_step, 0);

    READWRITE_PROPERTY(uint32_t, batch_size, NeRFConstants::batch_size);

private:
    cudaStream_t stream;
    std::unique_ptr<NeRFTrainingController> controller;
    
    uint32_t batch_size;
    uint32_t current_step;
    
public:
    // constructor

    Trainer() : mgr(std::make_unique<NeRFManager>(mgr)) {
        CUDA_CHECK_THROW(cudaStreamCreate(&this->stream));
    }

    // convenience funcs

    void load_dataset(const std::string& dataset_path) {
        Dataset dataset(dataset_path);

        auto nerf = this->mgr->create_trainable_nerf(this->stream, dataset.bounding_box);

        this->controller.reset(new NeRFTrainingController(dataset, nerf));
    }

    ~Trainer() {
        CUDA_CHECK_THROW(cudaStreamDestroy(this->stream));
    }
};

NRC_NAMESPACE_END
