#pragma once

#include "../../utils/hexagon-grid.cuh"
#include "../ray-batch-coordinators/hexagonal-grid-ray-batch-coordinator.cuh"
#include "../ray-batch-coordinators/hexagonal-tile-ray-batch-coordinator.cuh"
#include "render-task-factory.cuh"

NRC_NAMESPACE_BEGIN

class HexagonalGridRenderTaskFactory : public RenderTaskFactory {
public:
    using RenderTaskFactory::RenderTaskFactory;

    bool can_preview() const override {
        return true;
    }

    // TODO: this needs some revision/optimization.  It could be better at optimizing hexagon tiling.
    std::vector<RenderTask> create_tasks(const RenderRequest* request) override {
        // aspect ratio of hexagon is constant and this is a good value for an aesthetic shape
        float a = 1.1f;

        // number of total pixels in the outptu image
        int n_pix_total = request->output->width * request->output->height;

        // number of pixels covered by each hexagon in the preview grid
        int p_npix = n_pix_total / n_rays_per_preview;

        // dimensional properties of preview hexagon
        int p_H = hex_height_for_npix_and_aspect(p_npix, a);
        
        // we want p_H to be the highest multiple of 10 where npix <= n_rays_per_task
        p_H = (std::max(p_H, 10) / 10) * 10;

        int p_W, p_cw;
        hex_get_W_and_cw(p_H, a, p_W, p_cw);

        // the full-res width should be the greatest positive even integer multiple of the preview height such that:
        // the number of pixels per hexagon is less than or equal to the number of rays per task
        // if no such multiple exists, then the full-res width is set equal to the preview width
        float sf = sqrtf((float)n_rays_per_task / (float)p_npix); // exact scale factor of full-res hexagon
        sf = floorf(sf / 2.0f) * 2.0f; // round down to nearest even integer
        sf = std::max(1.0f, sf); // ensure that s is at least 1

        // convert to integer
        int s = (int)sf;

        // dimensional properties of full-res hexagon
        int H = p_H * s;
        int W = p_W * s;
        int cw = p_cw * s;

        // recalculate the number of rays per task
        int n_rays = n_pix_total_in_hex(H, cw);

        // figure out how many preview hexagons we can fit in the image
        int pn_w = 1 + 2 * request->output->width / (p_W + p_cw);

        // we need pn_w to be odd
        pn_w = pn_w + (1 - pn_w & 1);

        int pn_h = 3 + request->output->height / p_H;

        // now figure out how many full-res hexagons
        int n_w = pn_w / s;
        int n_h = pn_h / s;
        int n_hex = n_w * n_h;

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
        hex_coords.reserve(n_hex);

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
        tasks.reserve(n_hex + 1);

        // the fist task is a hexagonal grid for a low-resolution preview.
        tasks.emplace_back(
            pn_w * pn_h,
            request->camera,
            request->proxies[0]->get_nerf_ptrs(),
            std::unique_ptr<RayBatchCoordinator>(
                new HexagonalGridRayBatchCoordinator(
                    { pn_w, pn_h },
                    { o.x + cw / 4, o.y }, // ???
                    p_W,
                    p_H,
                    p_cw
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
