#pragma once

#include "../../math/hexagon-grid.cuh"
#include "../ray-batch-coordinators/hexagonal-grid-ray-batch-coordinator.cuh"
#include "../ray-batch-coordinators/hexagonal-tile-ray-batch-coordinator.cuh"
#include "render-task-factory.cuh"

TURBO_NAMESPACE_BEGIN

class HexagonalGridRenderTaskFactory : public RenderTaskFactory {
public:
    using RenderTaskFactory::RenderTaskFactory;

    bool can_preview() const override {
        return true;
    }

    // TODO: this needs some revision/optimization.  It could be better at optimizing hexagon tiling.
    std::vector<RenderTask> create_tasks(const RenderRequest* request) override {
        // TODO: multi-gpu
        const int device_id = 0;

        // aspect ratio of hexagon is constant and this is a good value for an aesthetic shape
        float a = 1.1f;

        // number of total pixels in the outptu image
        int n_pix_total = request->output->width * request->output->height;

        // number of pixels covered by each hexagon in the preview grid
        int p_npix = n_pix_total / n_rays_per_preview;

        // dimensional properties of preview hexagon
        int p_H = hex_height_for_npix_and_aspect(p_npix, a);
        
        // we want p_H to be a multiple of M
        int M = 10; // for best results, M should be the smallest number such that a * M is a positive integer
        p_H = ((std::max(p_H, M) / M) + 1) * M;

        int p_W, p_cw;
        hex_get_W_and_cw(p_H, a, p_W, p_cw);

        // update p_npix to be the actual number of pixels in the preview hexagon
        p_npix = n_pix_total_in_hex(p_H, p_cw);

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

        // figure out how many preview hexagons we can fit in the image horizontally
        int pn_w = 1 + 2 * request->output->width / (p_W + p_cw);

        // we need pn_w to be odd
        pn_w = pn_w + (1 - pn_w & 1);

        // calculate number of preview hexagons in the vertical direction
        int pn_h = 2 + request->output->height / p_H;

        // we need pn_h to be odd
        pn_h = pn_h + (1 - pn_h & 1);

        // now figure out how many full-res hexagons will fit
        int n_w = std::max(3, pn_w / s) + 1;
        int n_h = std::max(3, pn_h / s) + 1;

        // n_w and n_h must be odd
        n_w = n_w + (1 - n_w & 1);
        n_h = n_h + (1 - n_h & 1);

        int n_hex = n_w * n_h;
        
        // find the position of the center preview hexagon
        int pci = pn_w / 2;
        int pcj = pn_h / 2;
        int phx, phy;
        hex_get_xy_from_ij(pci, pcj, p_H, p_W, p_cw, phx, phy);

        // center pixel coordinates
        int cx = request->output->width / 2;
        int cy = request->output->height / 2;

        // calculate preview grid offsets
        int2 po = {
            cx - phx + p_W / 2,
            cy - phy + p_H / 2
        };

        // find the position of the center full-res hexagon
        int ci = n_w / 2;
        int cj = n_h / 2;
        int hx, hy;
        hex_get_xy_from_ij(ci, cj, H, W, cw, hx, hy);
        
        // calculate full-res grid offsets
        int2 o = {
            cx - hx,
            cy - hy
        };

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
            device_id,
            pn_w * pn_h,
            request->camera,
            request->renderables,
            request->modifiers,
            std::unique_ptr<RayBatchCoordinator>(
                new HexagonalGridRayBatchCoordinator(
                    { pn_w, pn_h },
                    po,
                    p_W,
                    p_H,
                    p_cw
                )
            )
        );

        // next we create a task for each hexagonal tile to fill it in with higher detail.
        for (const auto& coords : hex_coords) {
            tasks.emplace_back(
                device_id,
                n_rays,
                request->camera,
                request->renderables,
                request->modifiers,
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

TURBO_NAMESPACE_END
