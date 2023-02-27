#pragma once

#include <glad/glad.h>
#include <memory>

#include "../controllers/nerf-rendering-controller.h"
#include "../models/camera.cuh"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../render-targets/cpu-render-buffer.cuh"
#include "../utils/queues.h"
#include "../common.h"

NRC_NAMESPACE_BEGIN

class BlenderRenderEngine 
{
private:
    NeRFRenderingController _renderer;
    CPURenderBuffer _render_target;
    std::function<void()> _request_redraw;

    TwoItemQueue _render_queue;
    DebounceQueue _draw_queue;

    GLuint _render_tex_id = 0;
    
    void enqueue_redraw() {
        _draw_queue.push([this]() {
            this->_render_target.synchronize();
            _request_redraw();
        });
    }
    
public:

    BlenderRenderEngine()
        : _renderer()
        , _render_queue()
        , _draw_queue(1)
    {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to load OpenGL with glad.");
        };
    };


    /** API METHODS */

    void set_request_redraw_callback(std::function<void()> callback) {
        this->_request_redraw = callback;
    }

    void request_render(const Camera& camera, std::vector<NeRFProxy*>& proxies, const RenderFlags& flags) {
        _renderer.cancel();

        _render_queue.push([this, camera, proxies, flags]() {
            auto request = std::make_shared<RenderRequest>(
                camera,
                proxies,
                &_render_target,
                flags,
                // on_complete
                [this]() {
                    this->enqueue_redraw();
                },
                // on_progress
                [this](float progress) {
                    this->enqueue_redraw();
                },
                // on_cancel
                [this]() {
                    // no-op
                }
            );

            _renderer.submit(request);
        });
    }

    void resize_render_surface(const uint32_t& width, const uint32_t& height) {
        _render_target.set_size(width, height);
    }

    // create or resize render texture
    void update_render_texture() {
        // if texture is not created, create it
        if (_render_tex_id == 0) {
            glGenTextures(1, &_render_tex_id);
            glBindTexture(GL_TEXTURE_2D, _render_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        }

        // fetch texture size
        int width, height;
        glBindTexture(GL_TEXTURE_2D, _render_tex_id);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);

        // if texture size is not correct, resize it
        if (width != _render_target.width || height != _render_target.height) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _render_target.width, _render_target.height, 0, GL_RGBA, GL_FLOAT, _render_target.get_rgba());
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _render_target.width, _render_target.height, GL_RGBA, GL_FLOAT, _render_target.get_rgba());
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }
    /**
     * The following functionality was adapted from the Pixar RenderMan for Blender plugin.
     * https://github.com/prman-pixar/RenderManForBlender/blob/main/display_driver/d_blender.cpp#L335
     * 
     * This will always be called from Blender's view_draw() function
     */

    void draw()
    {
        // For some reason these calls make rendering a little bit smoother
        // glFlush();
        // glFinish();

        update_render_texture();

        // copy render data
        // _renderer.write_to(&_render_target);

        // These are the vertices of a quad that covers the entire viewport.
        const float w = static_cast<float>(_render_target.width);
        const float h = static_cast<float>(_render_target.height);

        GLfloat vertices[8] = {
            0.0f, 0.0f,
               w, 0.0f,
               w,    h,
            0.0f,    h
        };

        GLfloat tex_coords[8] = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,
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

        // Upload render target to GPU
        glBindTexture(GL_TEXTURE_2D, _render_tex_id);
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
