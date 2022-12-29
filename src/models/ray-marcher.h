#pragma once

#include "../common.h"

#include "bounding-box.h"
#include "cascaded-occupancy-grid.cuh"

NRC_NAMESPACE_BEGIN

struct RayMarcher {
public:
	// constructor
	RayMarcher(
		const BoundingBox* bounding_box,
		const CascadedOccupancyGrid* occupancy_grid,
		const float& dt_min,
		const float& dt_max,
		const float& max_steps,
		const uint32_t* ray_idx,
		float* ray_t,
		bool* ray_alive,
		float* ray_o_x,
		float* ray_o_y,
		float* ray_o_z,
		float* ray_d_x,
		float* ray_d_y,
		float* ray_d_z,
		float* ray_r,
		float* ray_g,
		float* ray_b,
		float* ray_a,
	)
		: bounding_box(bounding_box)
		, occupancy_grid(occupancy_grid)
		, dt_min(dt_min)
		, dt_max(dt_max)
		, max_steps(max_steps)
		, ray_idx(ray_idx)
		, ray_t(ray_t)
		, ray_alive(ray_alive)
		, ray_o_x(ray_o_x)
		, ray_o_y(ray_o_y)
		, ray_o_z(ray_o_z)
		, ray_d_x(ray_d_x)
		, ray_d_y(ray_d_y)
		, ray_d_z(ray_d_z)
		, ray_r(ray_r)
		, ray_g(ray_g)
		, ray_b(ray_b)
		, ray_a(ray_a)
	{};

private:
	// properties
	const BoundingBox* bounding_box;
	const OccupancyGrid* occupancy_grid;
	const float& dt_min;
	const float& dt_max;
	const float& max_steps;
	const uint32_t* ray_idx;
	float* ray_t;
	bool* ray_alive;
	float* ray_o_x;
	float* ray_o_y;
	float* ray_o_z;
	float* ray_d_x;
	float* ray_d_y;
	float* ray_d_z;
	float* ray_r;
	float* ray_g;
	float* ray_b;
	float* ray_a;
	float* ray_t;
};

NRC_NAMESPACE_END
