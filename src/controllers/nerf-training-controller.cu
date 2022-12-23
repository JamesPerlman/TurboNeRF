
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <json/json.hpp>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/encoding.h>
#include <Eigen/Dense>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

#include <memory>
#include "nerf-training-controller.h"

using namespace nrc;
using namespace Eigen;
using namespace tcnn;
using namespace nlohmann;

NeRFTrainingController::NeRFTrainingController(
	Dataset& dataset,
	const uint32_t& num_layers,
	const uint32_t& hidden_dim,
	const uint32_t& geo_feat_dim,
	const uint32_t& num_layers_color,
	const uint32_t& hidden_dim_color
) {
	this->dataset = dataset;
	this->num_layers = num_layers;
	this->hidden_dim = hidden_dim;
	this->geo_feat_dim = geo_feat_dim;
	this->num_layers_color = num_layers_color;
	this->hidden_dim_color = hidden_dim_color;
	
	// TODO: set this properly based on the aabb
	double per_level_scale = 1.4472692012786865;

	// Create the Direction Encoding
	json direction_encoding_config = {
		{"otype", "SphericalHarmonics"},
		{"degree", 4},
	};
	
	direction_encoding = std::shared_ptr<Encoding<network_precision_t>>(
		tcnn::create_encoding<network_precision_t>(3, direction_encoding_config)
	);
	
	// Create the Density MLP
	json density_mlp_encoding_config = {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", per_level_scale},
	};
	
	json density_mlp_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", hidden_dim},
		{"n_hidden_layers", num_layers - 1},
	};

	density_mlp = std::shared_ptr<tcnn::cpp::Module>(
		tcnn::cpp::create_network_with_input_encoding(3, 1 + geo_feat_dim, density_mlp_encoding_config, density_mlp_network_config)
	);
	
	// Create the Color MLP
	
	uint32_t color_mlp_in_dim = direction_encoding->padded_output_width() + geo_feat_dim;
		
	json color_mlp_network_config = {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "Sigmoid"},
		{"n_neurons", hidden_dim_color},
		{"n_hidden_layers", num_layers_color - 1},
	};
	
	color_mlp = std::shared_ptr<tcnn::cpp::Module>(
		tcnn::cpp::create_network(color_mlp_in_dim, 3, color_mlp_network_config)
	);
	
	// Set up Optimizer
	
	json optimizer_config = {
		{"otype", "Adam"},
		{"learning_rate", 1e-2},
		{"epsilon", 1e-15},
	};
	
	optimizer = std::shared_ptr<Optimizer<tcnn::network_precision_t>>(
		create_optimizer<network_precision_t>(optimizer_config)
	);
	
	// RNG
	// todo: CURAND_ASSERT_SUCCESS
	curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandGenerateSeeds(rng);
}

NeRFTrainingController::~NeRFTrainingController() {
	curandDestroyGenerator(rng);
}

// Training data kernels

__global__ void stbi_uchar_to_float(
	const uint32_t n_elements,
	const stbi_uc* __restrict__ src,
	float* __restrict__ dst
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < n_elements) {
		dst[idx] = (float)src[idx] / 255.0f;
	}
}

__global__ void generate_training_image_indices(
	const uint32_t n_elements,
	const uint32_t n_images,
	uint32_t* __restrict__ image_indices
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= n_elements) return;
	
	image_indices[idx] = idx * n_images / n_elements;
}

__global__ void resize_floats_to_uint32_with_max(
	const uint32_t n_elements,
	const float* __restrict__ floats,
	uint32_t* __restrict__ uints,
	const float range_max
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= n_elements) return;
	
	float resized_val = floats[idx] * range_max;
	uints[idx] = (uint32_t)resized_val;
}

// generates rays and RGBs for training, assigns them to an array of contiguous data
__global__ void select_pixels_and_rays_from_training_data(
	const uint32_t n_batch_elements,
	const uint32_t n_images,
	const uint32_t image_data_stride,
	const Vector2i image_dimensions,
	const Camera* __restrict__ cameras,
	const stbi_uc* __restrict__ image_data,
	const uint32_t* __restrict__ pixel_indices,
	const uint32_t* __restrict__ image_indices,
	float* __restrict__ ray_rgbs,
	float* __restrict__ ray_dirs
) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_batch_elements) return;
	
	uint32_t image_idx = 0;// image_indices[idx];
	uint32_t pixel_idx = pixel_indices[idx];
	
	uint32_t pixel_x = pixel_idx % image_dimensions.x();
	uint32_t pixel_y = pixel_idx / image_dimensions.x();
	uint32_t x = pixel_x;
	uint32_t y = pixel_y;
	Camera cam = cameras[image_idx];
	
	uint32_t img_offset = image_idx * image_data_stride;
	Ray ray = cam.get_ray_at_pixel_xy(x, y);

	ray_dirs[0 * n_batch_elements + idx] = ray.d.x();
	ray_dirs[1 * n_batch_elements + idx] = ray.d.y();
	ray_dirs[2 * n_batch_elements + idx] = ray.d.z();

	stbi_uc r = image_data[img_offset + 3 * pixel_idx + 0];
	stbi_uc g = image_data[img_offset + 3 * pixel_idx + 1];
	stbi_uc b = image_data[img_offset + 3 * pixel_idx + 2];
	
	ray_rgbs[0 * n_batch_elements + idx] = (float)r / 255.0f;
	ray_rgbs[1 * n_batch_elements + idx] = (float)g / 255.0f;
	ray_rgbs[2 * n_batch_elements + idx] = (float)b / 255.0f;
}

// NeRFTrainingController member functions

void NeRFTrainingController::prepare_for_training(cudaStream_t stream, uint32_t batch_size) {
	// todo: init workspace from dataset?
	workspace.enlarge(stream, dataset.n_pixels_per_image, dataset.n_channels_per_image, dataset.images.size(), batch_size);
	// todo: manage cameras inside the workspace?
	workspace.cameras.resize_and_copy_from_host(dataset.cameras);
	load_images(stream);
}

void NeRFTrainingController::load_images(cudaStream_t stream) {
	// make sure images are all loaded into CPU and GPU
	uint32_t image_size = dataset.n_channels_per_image * dataset.n_pixels_per_image * sizeof(stbi_uc);
	dataset.load_images_in_parallel(
		[this, &image_size, &stream](const size_t& image_index, const TrainingImage& image) {
			if (image_index == 0) {
				cudaError_t error = cudaMemcpyAsync(
					workspace.image_data,
					image.data_cpu.get(),
					image_size,
					cudaMemcpyHostToDevice,
					stream
				);
				if (error != cudaSuccess) {
					printf("image error: %d", image_index);
				}
			}
		}
	);

	printf("All images loaded to GPU.\n");
}

void NeRFTrainingController::generate_next_training_batch(cudaStream_t stream, uint32_t training_step) {
	
	// next, pull rays from the dataset 
	
	curandStatus_t status = curandGenerateUniform(rng, workspace.random_floats, workspace.batch_size);
	if (status != CURAND_STATUS_SUCCESS) {
		printf("Error generating random floats for training batch.\n");
	}
	
	resize_floats_to_uint32_with_max<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size, workspace.random_floats, workspace.pixel_indices, dataset.n_pixels_per_image
	);

	// need a kernel that selects ray & pixel indices from the training images
	generate_training_image_indices<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		dataset.images.size(),
		workspace.image_indices
	);

	select_pixels_and_rays_from_training_data<<<n_blocks_linear(workspace.batch_size), n_threads_linear>>>(
		workspace.batch_size,
		dataset.images.size(),
		dataset.n_pixels_per_image * dataset.n_channels_per_image,
		dataset.image_dimensions,
		workspace.cameras.data(),
		workspace.image_data,
		workspace.pixel_indices,
		workspace.image_indices,
		workspace.rgb_batch,
		workspace.ray_dir_batch
	);
	
	// debug code (check indices are random)
	vector<uint32_t> random_indices_host(workspace.batch_size);
	CUDA_CHECK_THROW(cudaMemcpyAsync(random_indices_host.data(), workspace.pixel_indices, workspace.batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

	vector<float> pixel_batch_host(3 * 1024);
	CUDA_CHECK_THROW(cudaMemcpyAsync(pixel_batch_host.data(), workspace.rgb_batch, 3 * 1024 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	
	vector<stbi_uc> first_img_host(800 * 800 * 4);
	CUDA_CHECK_THROW(cudaMemcpyAsync(first_img_host.data(), workspace.image_data, 800 * 800 * 4 * sizeof(stbi_uc), cudaMemcpyDeviceToHost, stream));
	
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	uint32_t n_nonzero_elements = 0;
	for (const auto& f : first_img_host) {
		if (f > 0.0001f) {
			++n_nonzero_elements;
		}
	}
	printf("%d\n", n_nonzero_elements);

	vector<stbi_uc> random_pixels(800 * 800 * 3);
	for (const auto& i : random_indices_host) {
		random_pixels[3 * i + 0] = stbi_uc(first_img_host[3 * i + 0] * 255.0f);
		random_pixels[3 * i + 1] = stbi_uc(first_img_host[3 * i + 1] * 255.0f);
		random_pixels[3 * i + 2] = stbi_uc(first_img_host[3 * i + 2] * 255.0f);
	}

	int w = 800;
	int h = 800;
	int c;
	auto x = stbi_failure_reason();
	std::string filename = "C:\\Users\\bizon\\Developer\\NeRFRenderCore\\random_px_step_1" + std::to_string(training_step) + ".png";

	auto data_cpu = stbi_loadf("E:\\2022\\nerf-library\\testdata\\lego\\train\\r_0.png", &w, &h, &c, 4);
	
	std::vector<stbi_uc> im_uc(w * h * c);
	for (uint32_t i = 0; i < w; ++i) {
		for (uint32_t j = 0; j < h; ++j) {
			for (uint32_t k = 0; k < c; ++k) {
				uint32_t idx = (i * w + j) * c + k;
				im_uc[idx] = (stbi_uc)(first_img_host[idx] * 255.f);
			}
		}
	}
	auto xy = stbi_failure_reason();
	stbi_write_png(filename.c_str(), w, h, c, first_img_host.data(), w * c * sizeof(stbi_uc));
}

void NeRFTrainingController::train_step(cudaStream_t stream) {
	
	// Train the model (batch_size must be a multiple of tcnn::batch_size_granularity)
	uint32_t batch_size = tcnn::next_multiple((uint32_t)1000, tcnn::batch_size_granularity);
	for (int i = 0; i < 1; ++i) {
		generate_next_training_batch(stream, i);
	}
	/*
	GPUMatrix<float> network_input(workspace.network_input);
	GPUMatrix<float> network_output(workspace.network_output);

	for (int i = 0; i < n_training_steps; ++i) {
		generate_training_batch(&training_batch_inputs, &training_batch_targets); // <-- your code

		float loss;
		model.trainer->training_step(training_stream, training_batch_inputs, training_batch_targets);
		std::cout << "iteration=" << i << " loss=" << loss << std::endl;
	}

	// Use the model
	GPUMatrix<float> inference_inputs(n_input_dims, batch_size);
	generate_inputs(&inference_inputs); // <-- your code

	GPUMatrix<float> inference_outputs(n_output_dims, batch_size);
	model.network->inference(inference_inputs, inference_outputs);
	*/
}