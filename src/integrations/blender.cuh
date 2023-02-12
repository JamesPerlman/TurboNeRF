#pragma once


#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "../render-targets/opengl-render-surface.cuh"
#include "../common.h"

NRC_NAMESPACE_BEGIN

class BlenderRenderEngine 
{
private:
    BlenderRenderEngine() {}
    BlenderRenderEngine(BlenderRenderEngine const&) = delete;
    void operator=(BlenderRenderEngine const&) = delete;

    bool is_initialized = false;



    static BlenderRenderEngine& _getInstance()
    {
        static BlenderRenderEngine instance;
        return instance;
    }
    
    void _init()
    {
        if (!is_initialized)
        {
            if (glewInit() != GLEW_OK)
                throw std::runtime_error("Failed to initialize GLEW.");

            is_initialized = true;
        }
    }

public:

    static void init()
    {
        _getInstance()._init();
    }

    /**
     * The following functionality was adapted from the Pixar RenderMan for Blender plugin.
     * https://github.com/prman-pixar/RenderManForBlender/blob/main/display_driver/d_blender.cpp#L335
     * 
     */

    static void draw(OpenGLRenderSurface* render_surface)
    {
        // AFAIK this is completely undocumented by Blender.
        // It allows us to draw into a Blender viewport, keeping all data on the GPU.

        // These are the vertices of a quad that covers the entire viewport.
        const float w = static_cast<float>(render_surface->width);
        const float h = static_cast<float>(render_surface->height);

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
        
        //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, render_surface->width, render_surface->height, GL_RGBA, GL_FLOAT, NULL);

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
        glBindTexture(GL_TEXTURE_2D, render_surface->get_texture_id());
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
