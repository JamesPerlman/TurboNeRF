#pragma once


#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "../render-targets/render-target.cuh"
#include "../common.h"

NRC_NAMESPACE_BEGIN

class BlenderRenderEngine 
{
private:
    BlenderRenderEngine() {}
    BlenderRenderEngine(BlenderRenderEngine const&) = delete;
    void operator=(BlenderRenderEngine const&) = delete;

    bool is_initialized = false;

    GLfloat tex_coords[8] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };
    
    static BlenderRenderEngine& _getInstance()
    {
        static BlenderRenderEngine instance;
        return instance;
    }

public:

    static void bl_draw(RenderTarget& buffer)
    {
        // THANK YOU SO MUCH PIXAR I LOVE YOU
        // https://github.com/prman-pixar/RenderManForBlender/blob/main/display_driver/d_blender.cpp#L335

        // AFAIK this is completely undocumented.  It allows us to draw into a Blender viewport, keeping all data on the GPU.

        GLint shader_program_id;
        glGetIntegerv(GL_CURRENT_PROGRAM, &shader_program_id);

        GLuint vertex_array;
        glGenVertexArrays(1, &vertex_array);
        glBindVertexArray(vertex_array);

        GLuint texcoord_location = glGetAttribLocation(shader_program_id, "texCoord");
        GLuint position_location = glGetAttribLocation(shader_program_id, "pos");

        glEnableVertexAttribArray(texcoord_location);
        glEnableVertexAttribArray(position_location);

    }

};

NRC_NAMESPACE_END
