#pragma once

#include <chrono>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <memory>
#include <mutex>
#include <tuple>
#include <rxcpp/rx.hpp>

#include "../controllers/nerf-rendering-controller.h"
#include "../models/camera.cuh"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../render-targets/opengl-render-surface.cuh"
#include "../common.h"

NRC_NAMESPACE_BEGIN

class BlenderRenderEngine 
{
private:
    NeRFRenderingController _renderer;
    std::mutex _drawing_mutex;
    OpenGLRenderSurface _render_surface;
    std::function<void()> _tag_redraw;

    rxcpp::subjects::subject<RenderRequest> rx_draw_request;
    rxcpp::subjects::subject<RenderRequest> rx_render_request;
    rxcpp::subjects::subject<RenderRequest> rx_render_finish;

private:
    void rx_init() {
        /**
         * rx_draw_request needs to be debounced to preserve performance.
         */
        rx_draw_request
            .get_observable()
            .debounce(std::chrono::milliseconds(333))
            .subscribe([this](RenderRequest request) {
                this->request_redraw();
            });
        
        /**
         * rx_render_request is a stream of RenderRequest objects
         * This subscriber launches a new render if we are not currently rendering.
         */

        rx_render_request
            .get_observable()
            .subscribe([this](RenderRequest request) {
                if (!_renderer.is_rendering()) {
                    _renderer.submit(request, true);
                }
            });

        /**
         * rx_render_finish is a stream of RenderRequest objects that have either canceled or completed
         * 
         * Here we subscribe to the finished requests, and if a new request is available we start a new render.
         */

        // rx_render_finish
        //     .get_observable()
        //     .with_latest_from(rx_render_request.get_observable())
        //     .filter([this](std::tuple<RenderRequest, RenderRequest> v) {
        //         return std::get<0>(v) != std::get<1>(v);
        //     })
        //     .subscribe([this](std::tuple<RenderRequest, RenderRequest> v) {
        //         RenderRequest next_request = std::get<1>(v);
        //         _renderer.submit(next_request, true);
        //     });
        auto o1 = rxcpp::observable<>::interval(std::chrono::milliseconds(2));
        auto o2 = rxcpp::observable<>::interval(std::chrono::milliseconds(3));
        auto o3 = rxcpp::observable<>::interval(std::chrono::milliseconds(5));
        auto values = o1.with_latest_from(o2, o3);
        values.
            take(5).
            subscribe(
                [](std::tuple<int, int, int> v){printf("OnNext: %d, %d, %d\n", std::get<0>(v), std::get<1>(v), std::get<2>(v));},
                [](){printf("OnCompleted\n");});
    }

public:

    BlenderRenderEngine()
        : _renderer()
    {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to load OpenGL with glad.");
        };
        rx_init();
    };


    void request_redraw() {
        _tag_redraw();
    }

    /** API METHODS */

    void set_tag_redraw_callback(std::function<void()> tag_redraw) {
        this->_tag_redraw = tag_redraw;
    }

    void request_render(const Camera& camera, std::vector<NeRFProxy*>& proxies) {
        RenderRequest request{
            camera,
            proxies,
            &_render_surface,
            // on_complete
            [this](RenderRequest& request) {
                this->rx_render_finish
                    .get_subscriber()
                    .on_next(request);
                printf("Render request completed!\n");
            },
            // on_progress
            [this](RenderRequest& request) {
                this->rx_draw_request
                    .get_subscriber()
                    .on_next(request);
            },
            // on_cancel
            [this](RenderRequest& request) {
                this->rx_render_finish
                    .get_subscriber()
                    .on_next(request);
                printf("Render request cancelled!\n");
            }
        };

        rx_render_request
            .get_subscriber()
            .on_next(request);
    }

    void resize_render_surface(const uint32_t& width, const uint32_t& height) {
        _render_surface.set_size(width, height);
    }

    /**
     * The following functionality was adapted from the Pixar RenderMan for Blender plugin.
     * https://github.com/prman-pixar/RenderManForBlender/blob/main/display_driver/d_blender.cpp#L335
     * 
     * This needs to be called from the view_draw() method of the Blender RenderEngine.
     * view_draw does some sneaky OpenGL stuff to get the Context ready for drawing.
     * I have not quite figured out how to get the Context ready on a background thread.
     * Serious speed up achievable if we can figure this out.
     * 
     */

    void draw()
    {
        std::scoped_lock lock(_drawing_mutex);
        
        // copy render data
        // if this can be moved to a background thread that would be ideal.
        // Or maybe we only write the pixels that have changed?
        _renderer.write_to(&_render_surface);

        // From here on it's just OpenGL drawing code to get the rendered data onscreen.

        // These are the vertices of a quad that covers the entire viewport.
        const float w = static_cast<float>(_render_surface.width);
        const float h = static_cast<float>(_render_surface.height);

        GLfloat vertices[8] = {
            0.0f, 0.0f,
               w, 0.0f,
               w,    h,
            0.0f,    h
        };

        GLfloat tex_coords[8] = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
            0.0f, 1.0f
        };
        
        GLint shader_program_id;
        glGetIntegerv(GL_CURRENT_PROGRAM, &shader_program_id);

        GLuint vertex_array;
        glGenVertexArrays(1, &vertex_array);
        glBindVertexArray(vertex_array);

        // Get the location of the "texCoord" and "pos" attributes in the shader program
        GLuint texcoord_location = glGetAttribLocation(shader_program_id, "texCoord");
        GLuint position_location = glGetAttribLocation(shader_program_id, "pos");

        glEnableVertexAttribArray(texcoord_location);
        glEnableVertexAttribArray(position_location);

        // Generate a vertex buffer object for the texture coordinates
        GLuint texture_vbo_id;
        glGenBuffers(1, &texture_vbo_id);
        glBindBuffer(GL_ARRAY_BUFFER, texture_vbo_id);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), &tex_coords[0], GL_STATIC_DRAW);
        glVertexAttribPointer(texcoord_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), reinterpret_cast<void*>(0));

        GLuint vertex_vbo_id;
        // Generate a vertex buffer object for the positions
        glGenBuffers(1, &vertex_vbo_id);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo_id);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

        // Enable the "pos" attribute
        glEnableVertexAttribArray(position_location);
        glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), reinterpret_cast<void*>(0));

        glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind the current buffer
        glBindVertexArray(0); // Unbind the vertex array object

        // Activate texture unit 0
        glActiveTexture(GL_TEXTURE0);

        // Bind to render surface texture
        glBindTexture(GL_TEXTURE_2D, _render_surface.get_texture_id());
        glBindVertexArray(vertex_array);

        // Draw the triangle fan
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glBindVertexArray(0); // Unbind the vertex array object
        glDeleteVertexArrays(1, &vertex_array); // Delete the vertex array object
        glBindTexture(GL_TEXTURE_2D, 0); // Unbind the texture

        // Clean up

        // Delete the vertex array object
        glDeleteVertexArrays(1, &vertex_array);
        // Delete the vertex buffer objects
        glDeleteBuffers(1, &vertex_vbo_id);
        glDeleteBuffers(1, &texture_vbo_id);
    }
};

NRC_NAMESPACE_END
