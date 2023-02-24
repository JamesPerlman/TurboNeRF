#pragma once

#include "../../utils/hexagon-grid.cuh"
#include "../ray-batch-coordinators/hexagonal-grid-ray-batch-coordinator.cuh"
#include "../ray-batch-coordinators/hexagonal-tile-ray-batch-coordinator.cuh"
#include "render-task-factory.cuh"

NRC_NAMESPACE_BEGIN

class HexagonalGridRenderTaskFactory : public RenderTaskFactory {
public:
    HexagonalGridRenderTaskFactory(
        const int& n_rays_per_task,
        const int& n_tasks_per_batch
    ) : RenderTaskFactory(n_rays_per_task, n_tasks_per_batch) {}


    // TODO: this needs some revision/optimization.  It could be better at optimizing hexagon tiling.
    std::vector<RenderTask> create_tasks(const RenderRequest* request) override {
        // determine how large each hexagon is
        float a = 1.1f; // aspect ratio of hexagon
        int H = hex_height_for_npix_and_aspect(n_rays_per_task, a);

        // we want H to be the highest multiple of 40 where npix <= n_rays_per_task
        H = (std::max(H, 40) / 40) * 40;

        // get width of hexagon and width of rectangular central region
        int W, cw;
        hex_get_W_and_cw(H, a, W, cw);

        // recalculate the number of rays per task
        int n_rays = n_pix_total_in_hex(H, cw);

        // figure out how many hexagons we can fit in the image
        int n_w = 1 + 2 * request->output->width / (W + cw);

        // we need n_w to be odd
        n_w = n_w + (1 - n_w & 1);

        int n_h = 3 + request->output->height / H;
        int n_hexagons = n_w * n_h;

        // center pixel coordinates
        int cx = request->output->width / 2;
        int cy = request->output->height / 2;

        // find the position of the center hexagon
        int ci = n_w / 2;
        int cj = n_h / 2;
        int hx, hy;
        hex_get_xy_from_ij(ci, cj, H, W, cw, hx, hy);
        
        // calculate grid offsets
        int2 o = { cx - hx, cy - hy };

        // prepare to create tasks by order of distance from the center
        std::vector<int2> hex_coords;
        hex_coords.reserve(n_hexagons);

        for (int i = 0; i < n_w; ++i) {
            for (int j = 0; j < n_h; ++j) {
                int x, y;
                hex_get_xy_from_ij(i, j, H, W, cw, x, y);
                hex_coords.push_back({
                    x + o.x,
                    y + o.y
                });
            }
        }

        // sort by distance from center
        std::sort(
            hex_coords.begin(),
            hex_coords.end(),
            [&cx, &cy](const int2& a, const int2& b) {
                int ax = a.x - cx;
                int ay = a.y - cy;
                int bx = b.x - cx;
                int by = b.y - cy;

                return (ax * ax + ay * ay) < (bx * bx + by * by);
            }
        );

        // create tasks
        std::vector<RenderTask> tasks;
        tasks.reserve(1);

        // the fist task is a hexagonal grid for a low-resolution preview.
        tasks.emplace_back(
            n_w * n_h,
            request->camera,
            request->proxies[0]->get_nerf_ptrs(),
            std::unique_ptr<RayBatchCoordinator>(
                new HexagonalGridRayBatchCoordinator(
                    { n_w, n_h },
                    { o.x + cw / 2, o.y }, // ???
                    W,
                    H,
                    cw
                )
            )
        );

        // next we create a task for each hexagonal tile to fill it in with higher detail.
        for (const auto& coords : hex_coords) {
            tasks.emplace_back(
                n_rays,
                request->camera,
                request->proxies[0]->get_nerf_ptrs(),
                std::unique_ptr<RayBatchCoordinator>(
                    new HexagonalTileRayBatchCoordinator(
                        n_rays,
                        W,
                        H,
                        cw,
                        coords.x - W / 2,
                        coords.y - H / 2
                    )
                )
            );
        }

        return tasks;
    }
};

NRC_NAMESPACE_END
