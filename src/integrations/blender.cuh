#pragma once

#include <glad/glad.h>
#include <memory>
#include <mutex>

#include "../controllers/nerf-rendering-controller.h"
#include "../models/camera.cuh"
#include "../models/nerf-proxy.cuh"
#include "../models/render-request.cuh"
#include "../render-targets/opengl-render-surface.cuh"
#include "../utils/queues.h"
#include "../common.h"

NRC_NAMESPACE_BEGIN

class BlenderRenderEngine 
{
private:
    NeRFRenderingController _renderer;
    OpenGLRenderSurface _render_surface;
    std::function<void()> _tag_redraw;

    TwoItemQueue _render_queue;
    DebounceQueue _draw_queue;
    
    void request_redraw() {
        _draw_queue.push([this]() {
            _tag_redraw();
        });
    }
    
public:

    BlenderRenderEngine()
        : _renderer()
        , _render_queue()
        , _draw_queue(333)
    {
        if (!gladLoadGL()) {
            throw std::runtime_error("Failed to load OpenGL with glad.");
        };
    };


    /** API METHODS */

    void set_tag_redraw_callback(std::function<void()> tag_redraw) {
        this->_tag_redraw = tag_redraw;
    }

    void request_render(const Camera& camera, std::vector<NeRFProxy*>& proxies) {
        _renderer.cancel();

        _render_queue.push([this, camera, proxies]() {
            auto request = std::make_shared<RenderRequest>(
                camera,
                proxies,
                &_render_surface,
                // on_complete
                [this]() {
                    this->request_redraw();
                    printf("Render complete!\n");
                },
                // on_progress
                [this](float progress) {
                    this->request_redraw();
                    printf("Render progress: %f\n", progress);
                },
                // on_cancel
                [this]() {
                    printf("Render request cancelled!\n");
                }
            );

            _renderer.submit(request);
        });
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
