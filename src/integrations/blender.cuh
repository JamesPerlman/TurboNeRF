#pragma once


#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "../models/render-buffer.cuh"
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

    static void bl_draw(RenderBuffer& buffer)
    {

        GLint shader_program_id;
        glGetIntegerv(GL_CURRENT_PROGRAM, &shader_program_id);
        GLubyte x = 0;
    }

};

NRC_NAMESPACE_END
