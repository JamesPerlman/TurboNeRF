#pragma once

#include <chrono>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <memory>
#include <mutex>

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
    bool _is_drawing = false;
    bool _did_request_redraw = false;
    int _current_draw_request_id = 0;
    int _n_draw_requests = 0;
    std::mutex _drawing_mutex;
    OpenGLRenderSurface _render_surface;
    std::unique_ptr<RenderRequest> _current_request;
    std::unique_ptr<RenderRequest> _next_request;
    std::function<void()> _tag_redraw;
    std::future<void> _render_flusher;
    std::future<void> _redraw_future;

    std::chrono::steady_clock::time_point _last_draw_time = std::chrono::steady_clock::now();

public:

    BlenderRenderEngine()
        : _renderer()
    {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to load OpenGL with glad.");
        };
    };


    /** INTERNAL METHODS **/
    // this has a property of throttle_time
    void submit_draw_request(const std::chrono::milliseconds& throttle_duration) {
        if (throttle_duration > std::chrono::seconds(0) && std::chrono::high_resolution_clock::now() - _last_draw_time < throttle_duration) {
            return;
        }

        if (_redraw_future.valid() && _redraw_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return;
        }

        _redraw_future = std::async(
            std::launch::async,
            [this]() {
                std::unique_lock lock(_drawing_mutex);
                ++_n_draw_requests;
                bool is_drawing = _is_drawing;
                lock.unlock();

                // wait for _is_drawing to be false
                while (is_drawing) {
                    lock.lock();
                    is_drawing = _is_drawing;
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                
                request_redraw();
            }
        );
    }

    void submit_render_request() {
        if (_renderer.is_rendering()) {
            return;
        }
        _renderer.submit(_current_request.get(), true);
        _render_flusher = std::async(
            std::launch::async,
            [this]() {
                _renderer.wait_until_finished();
                this->flush_render_queue();
            }
        );
    }

    void flush_render_queue() {
        _current_request.reset();
        _current_request = nullptr;
        if (_next_request != nullptr) {
            _current_request = std::move(_next_request);
            _next_request.reset();
            submit_render_request();
        }
    }

    void request_redraw() {
        std::unique_lock lock(_drawing_mutex);
        
        if (_did_request_redraw)
            return;

        _did_request_redraw = true;
        
        lock.unlock();

        _tag_redraw();
    }

    /** API METHODS */

    void set_tag_redraw_callback(std::function<void()> tag_redraw) {
        this->_tag_redraw = tag_redraw;
    }

    void did_begin_drawing() {
        std::scoped_lock lock(_drawing_mutex);
        _is_drawing = true;
        _current_draw_request_id = _n_draw_requests;
    }

    void did_finish_drawing() {
        std::unique_lock lock(_drawing_mutex);
        _is_drawing = false;
        _did_request_redraw = false;

        // Is there still a draw request queued up?
        if (_current_draw_request_id < _n_draw_requests) {
            lock.unlock();
            // If so, then we need to draw again
            request_redraw();
        } else {
            // This is the last draw request, so we can reset the counter
            _n_draw_requests = 0;
        }

        _last_draw_time = std::chrono::high_resolution_clock::now();
    }

    void request_render(const Camera& camera, std::vector<NeRFProxy*>& proxies) {
        RenderRequest request{
            camera,
            proxies,
            &_render_surface,
            // on_complete
            [this]() {
                this->submit_draw_request(std::chrono::milliseconds(0));
                printf("Render request complete!\n");
            },
            // on_progress
            [this](float progress) {
                // if we are already drawing, then we need to queue up another draw request
                this->submit_draw_request(std::chrono::milliseconds(333));
            },
            // on_cancel
            [this]() {
                printf("Render request cancelled!\n");
            }
        };

        if (_current_request == nullptr) {
            _current_request = std::make_unique<RenderRequest>(request);
        } else {
            _current_request->cancel();
            _next_request.reset();
            _next_request = std::make_unique<RenderRequest>(request);
        }

        submit_render_request();
    }

    void resize_render_surface(const uint32_t& width, const uint32_t& height) {
        _render_surface.set_size(width, height);
    }

    /**
     * The following functionality was adapted from the Pixar RenderMan for Blender plugin.
     * https://github.com/prman-pixar/RenderManForBlender/blob/main/display_driver/d_blender.cpp#L335
     * 
     * This will always be called from Blender's view_draw() function
     */

    void draw()
    {
        std::scoped_lock lock(_drawing_mutex);
        
        // copy render data
        _renderer.write_to(&_render_surface);

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
