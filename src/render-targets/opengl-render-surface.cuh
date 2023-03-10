#pragma once

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "render-target.cuh"

// Adapted from https://www.informit.com/articles/article.aspx?p=2455391&seqNum=2 - thank you informIT!

TURBO_NAMESPACE_BEGIN

class OpenGLRenderSurface: public RenderTarget {
private:
    GLuint pbo;
    GLuint texture;
    cudaGraphicsResource* cuda_pbo_res;

    void allocate(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {
        
        // get previous state
        GLint prev_buffer, prev_texture;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev_buffer);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_texture);

        // allocate

        this->width = width;
        this->height = height;

        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

        CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_pbo_res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

        // reset to previous state
        glBindTexture(GL_TEXTURE_2D, prev_texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, prev_buffer);
    }

    void resize(const uint32_t& width, const uint32_t& height, const cudaStream_t& stream = 0) override {

        // get previous state
        GLint prev_buffer, prev_texture;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev_buffer);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_texture);

        // resize

        this->width = width;
        this->height = height;

        CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(cuda_pbo_res));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

        CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_pbo_res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
        
        // reset to previous state
        glBindTexture(GL_TEXTURE_2D, prev_texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, prev_buffer);
    }
    
public:

    using RenderTarget::RenderTarget;

    GLuint get_texture_id() const {
        return texture;
    }

    void free(const cudaStream_t& stream = 0) override {
        if (width == 0 || height == 0)
            return;
        
        cudaGraphicsUnregisterResource(cuda_pbo_res);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture);
    }

    void open_for_cuda_access(std::function<void(float* rgba)> handle, const cudaStream_t& stream = 0) override {

        // get previously bound buffer
        GLint prev_buffer;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev_buffer);

        // previous texture
        GLint prev_texture;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_texture);

        // prepare for cuda access
        float* rgba;
        CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &cuda_pbo_res, stream));
        CUDA_CHECK_THROW(cudaGraphicsResourceGetMappedPointer((void **)&rgba, nullptr, cuda_pbo_res));

        handle(rgba);

        CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &cuda_pbo_res, stream));

        // upload to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);

        // restore previous state
        glBindTexture(GL_TEXTURE_2D, prev_texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, prev_buffer);
    }
};

TURBO_NAMESPACE_END


/**
 * Some discussion on the "stickiness" of the UI.
 * 
 * I'm trying to figure this out.  For some reason there is lag when mapping/unmapping any opengl resource.
 * I tried mapping a texture instead of a pixel buffer, and I tried mapping an array and writing to a surface,
 * all to no avail.
 * 
 * cudaCreateTextureObject? cudaCreateSurfaceObject? surf2Dwrite? For now we will just write to the pixel buffer.
 * I suspect the lag might be coming from a roundtrip to the CPU, but I'm not sure where it is.
 * 
 * Research:
 * https://www.informit.com/articles/article.aspx?p=2455391&seqNum=2
 * https://github.dev/ndd314/cuda_examples/blob/master/3_Imaging/postProcessGL
 * https://github.dev/Hello100blog/gl_cuda_interop_pingpong_st
 * https://stackoverflow.com/questions/15053444/modifying-opengl-fbo-texture-attachment-in-cuda
 * https://forums.developer.nvidia.com/t/cudabindtexturetoarray-deprecated/176713/
 * 
 * Update: it is fixed. https://github.com/JamesPerlman/TurboNeRF/discussions/16#discussioncomment-5095071
 */
